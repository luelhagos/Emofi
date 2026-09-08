import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio

random.seed(1234)


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = os.path.expandvars(
            os.path.expanduser(config["path"]["raw_path"])
        )
        self.out_dir = os.path.expandvars(
            os.path.expanduser(config["path"]["preprocessed_path"])
        )
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        # Keep up to this much leading/trailing silence as an explicit sp token
        # (0 = trim at the first/last speech phone). Training targets that end
        # exactly on the final phone's low-energy offset teach the decoder an
        # end-of-sequence energy roll-off; a short terminal silence token gives
        # that behavior a correct place to land.
        self.boundary_sil_sec = config["preprocessing"].get(
            "boundary_silence_max_sec", 0.0
        )

        # Lexicon for recovering spn-aligned words. MFA emits a single 'spn'
        # phone when it cannot fit a word's phones (clipped/degraded audio);
        # treating spn as silence then deletes the word from the features and
        # manifests. When the words tier names the word and the lexicon has
        # it, we substitute the word's phones across the spn interval instead.
        self.lexicon = {}
        lex_path = os.path.expandvars(
            os.path.expanduser(config["path"].get("lexicon_path", ""))
        )
        if lex_path and os.path.exists(lex_path):
            with open(lex_path, encoding="utf-8") as f:
                for line in f:
                    t = line.strip().split()
                    if len(t) >= 2 and t[0].lower() not in self.lexicon:
                        self.lexicon[t[0].lower()] = t[1:]
        self.spn_fixed = 0
        self.spn_kept = 0

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        done = reused = 0

        # Feature extraction. Resumable: utterances whose feature files already
        # exist are skipped (only their metadata line is rebuilt from the
        # TextGrid). Normalization is deferred to the finalize step below, so a
        # killed run can simply be re-run until extraction completes.
        speakers = {}
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                chapter = basename.split("_")[1]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, chapter, "{}.TextGrid".format(basename)
                )
                if not os.path.exists(tg_path):
                    continue

                if self._features_exist(speaker, basename):
                    line = self._metadata_line(speaker, chapter, basename)
                    if line is not None:
                        out.append(line)
                        reused += 1
                    continue

                ret = self.process_utterance(speaker, chapter, basename)
                if ret is None:
                    continue
                out.append(ret[0])
                done += 1
        print(f"Extraction complete: {done} new, {reused} reused")
        print(f"spn recovery: {self.spn_fixed} words substituted from the "
              f"lexicon, {self.spn_kept} kept as spn (no word / not in lexicon)")

        # Stats + in-place normalization (runs once, guarded by stats.json).
        self._finalize_stats()

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def _features_exist(self, speaker, basename):
        for sub in ("duration", "pitch", "energy", "mel"):
            p = os.path.join(
                self.out_dir, sub, "{}-{}-{}.npy".format(speaker, sub, basename)
            )
            if not os.path.exists(p) or os.path.getsize(p) < 128:
                return False
        return True

    def _metadata_line(self, speaker, chapter, basename):
        """Rebuild the manifest line for an already-extracted utterance from
        its TextGrid and .lab file (no audio work)."""
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, chapter, "{}.TextGrid".format(basename)
        )
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        try:
            textgrid = tgt.io.read_textgrid(tg_path)
            try:
                words_tier = textgrid.get_tier_by_name("words")
            except Exception:
                words_tier = None
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones"), words_tier
            )
            if start >= end or not phone:
                return None
            with open(text_path, "r") as f:
                raw_text = f.readline().strip("\n")
            return "|".join(
                [basename, speaker, "{" + " ".join(phone) + "}", raw_text]
            )
        except Exception:
            return None

    def _finalize_stats(self):
        """Compute pitch/energy stats from the saved feature files, normalize
        them in place, and write stats.json. Runs once: if stats.json exists
        the features are already normalized and this is a no-op."""
        stats_path = os.path.join(self.out_dir, "stats.json")
        if os.path.exists(stats_path):
            print("stats.json exists -- features already normalized, skipping")
            return

        print("Computing statistic quantities ...")
        pitch_dir = os.path.join(self.out_dir, "pitch")
        energy_dir = os.path.join(self.out_dir, "energy")
        scalers = {"pitch": StandardScaler(), "energy": StandardScaler()}
        bad = []
        for name, d in (("pitch", pitch_dir), ("energy", energy_dir)):
            for fn in tqdm(sorted(os.listdir(d)), desc=f"stats:{name}"):
                try:
                    v = np.load(os.path.join(d, fn))
                except Exception:
                    bad.append(os.path.join(d, fn))
                    continue
                v = self.remove_outlier(v)
                if len(v) > 0:
                    scalers[name].partial_fit(v.reshape((-1, 1)))
        if bad:
            for p in bad:
                print(f"corrupt feature file: {p}")
            raise SystemExit(
                "Delete the corrupt files above and re-run (their utterances "
                "will be re-extracted)."
            )

        if self.pitch_normalization:
            pitch_mean = scalers["pitch"].mean_[0]
            pitch_std = scalers["pitch"].scale_[0]
        else:
            pitch_mean, pitch_std = 0, 1
        if self.energy_normalization:
            energy_mean = scalers["energy"].mean_[0]
            energy_std = scalers["energy"].scale_[0]
        else:
            energy_mean, energy_std = 0, 1

        pitch_min, pitch_max = self.normalize(pitch_dir, pitch_mean, pitch_std)
        energy_min, energy_max = self.normalize(energy_dir, energy_mean, energy_std)

        mel_dir = os.path.join(self.out_dir, "mel")
        mel_min, mel_max = float("inf"), -float("inf")
        n_frames = 0
        for fn in tqdm(sorted(os.listdir(mel_dir)), desc="stats:mel"):
            m = np.load(os.path.join(mel_dir, fn), mmap_mode="r")
            mel_min = min(mel_min, float(m.min()))
            mel_max = max(mel_max, float(m.max()))
            n_frames += m.shape[0]

        with open(stats_path, "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "mel": [
                    float(mel_min),
                    float(mel_max),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

    def process_utterance(self, speaker, chapter, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, chapter, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        try:
            words_tier = textgrid.get_tier_by_name("words")
        except Exception:
            words_tier = None
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones"), words_tier
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram),
            np.max(mel_spectrogram),
        )

    def get_alignment(self, tier, words_tier=None):
        sil_phones = ["sil", "sp", "spn"]

        # Recover spn-aligned words. A pre-pass expands any 'spn' phone whose
        # span carries a real word (from the words tier) into that word's
        # lexicon phones, splitting the interval evenly. Without this the
        # silence handling below deletes the word (utterance-initial/final spn)
        # or reduces it to a pause token (internal spn).
        intervals = []
        words = []
        if words_tier is not None:
            words = [
                (w.start_time, w.end_time, w.text.strip())
                for w in words_tier._objects
                if w.text.strip()
            ]
        for t in tier._objects:
            s0, e0, p0 = t.start_time, t.end_time, t.text
            if p0 == "spn" and words and self.lexicon:
                mid = (s0 + e0) / 2
                wtext = next(
                    (wt for ws, we, wt in words if ws <= mid <= we), None
                )
                sub = self.lexicon.get(wtext.lower()) if wtext else None
                if sub:
                    step = (e0 - s0) / len(sub)
                    for j, ph in enumerate(sub):
                        intervals.append((s0 + j * step, s0 + (j + 1) * step, ph))
                    self.spn_fixed += 1
                    continue
                self.spn_kept += 1
            intervals.append((s0, e0, p0))

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        prev_e = None  # end time of previous interval, for gap detection
        for s, e, p in intervals:

            # MFA can leave gaps between intervals (unfilled pauses) instead of
            # explicit sp/sil intervals. Fill any internal gap with an 'sp' token
            # so the durations span the full timeline; leading gaps are handled
            # by the start-time trimming below.
            if prev_e is not None and (s - prev_e) > 1e-4 and phones != []:
                phones.append("sp")
                durations.append(
                    int(
                        np.round(s * self.sampling_rate / self.hop_length)
                        - np.round(prev_e * self.sampling_rate / self.hop_length)
                    )
                )
            prev_e = e

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Boundary silence. With boundary_sil_sec == 0, trim at the last speech
        # phone (original behavior). Otherwise keep ONE leading and ONE trailing
        # sp token, each capped, so utterances do not start/end exactly on a
        # speech phone.
        trail_frames = int(sum(durations[end_idx:]))
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        if self.boundary_sil_sec > 0 and phones:
            cap = int(
                round(self.boundary_sil_sec * self.sampling_rate / self.hop_length)
            )
            # trailing: keep up to `cap` frames of the silence after the last
            # speech phone (extends the audio span accordingly)
            keep = min(trail_frames, cap)
            if keep > 0:
                phones.append("sp")
                durations.append(keep)
                end_time = end_time + keep * self.hop_length / self.sampling_rate
            # leading: keep up to `cap` frames of audio before the first speech
            # phone (moves the span start earlier)
            lead_sec = min(start_time, cap * self.hop_length / self.sampling_rate)
            lead = int(
                round(start_time * self.sampling_rate / self.hop_length)
                - round(
                    (start_time - lead_sec) * self.sampling_rate / self.hop_length
                )
            )
            if lead > 0:
                phones.insert(0, "sp")
                durations.insert(0, lead)
                start_time = start_time - lead_sec

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

