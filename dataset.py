import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D, expand

random.seed(1234)


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False,
        speaker_emb=False, mismatch_face=False, mismatch_face_prob=0.5,
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text, self.speaker_to_ids = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.face_emb_dir = os.path.join(self.preprocessed_path, "face_emb_cache")
        # ECAPA speaker-embedding cache (parallel to face_emb).
        self.speaker_emb = speaker_emb
        self.spk_emb_dir = os.path.join(self.preprocessed_path, "spk_emb_cache")

        # Cross-speaker face sampling. With probability mismatch_face_prob the
        # face embedding comes from a different actor's utterance with the same
        # (emotion, intensity level); otherwise from a different utterance of the
        # same actor in that cell. The speaker embedding always stays the
        # target's own.
        self.mismatch_face = mismatch_face
        self.mismatch_face_prob = mismatch_face_prob
        if mismatch_face:
            self.emo_int_ids = {}   # (emotion, level) -> [dataset index]
            for i, b in enumerate(self.basename):
                parts = b.split("_")
                self.emo_int_ids.setdefault((parts[1], parts[2]), []).append(i)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        query_idx = random.choice(self.speaker_to_ids[speaker]) # Sample the query text
        raw_quary_text = self.raw_text[query_idx]
        query_phone = np.array(text_to_sequence(self.text[query_idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        quary_duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(self.speaker[query_idx], self.basename[query_idx]),
        )
        quary_duration = np.load(quary_duration_path)

        # Face source: the target's own face unless mismatch_face is on. Falls
        # back to the target's own face if the sampled pool is empty.
        face_base = basename
        if self.mismatch_face:
            parts = basename.split("_")
            cell = self.emo_int_ids.get((parts[1], parts[2]), [])
            if random.random() < self.mismatch_face_prob:
                pool = [j for j in cell if self.speaker[j] != speaker]
            else:
                pool = [j for j in cell
                        if self.speaker[j] == speaker and self.basename[j] != basename]
            if pool:
                face_base = self.basename[random.choice(pool)]
        face_emb = torch.load(
            os.path.join(self.face_emb_dir, "{}.pt".format(face_base)),
            map_location="cpu",
        ).numpy()  # (768,)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "quary_text": query_phone,
            "raw_quary_text": raw_quary_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "quary_duration": quary_duration,
            "face_emb": face_emb,
        }

        # ECAPA speaker embedding for the speaker channel (the utterance's own speaker).
        if self.speaker_emb:
            sample["spk_emb"] = torch.load(
                os.path.join(self.spk_emb_dir, "{}.pt".format(basename)),
                map_location="cpu",
            ).numpy()  # (192,)

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            speaker_to_ids = dict()
            for i, line in enumerate(f.readlines()):
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                if s not in speaker_to_ids:
                    speaker_to_ids[s] = [i]
                else:
                    speaker_to_ids[s] += [i]
            return name, speaker, text, raw_text, speaker_to_ids

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        quary_texts = [data[idx]["quary_text"] for idx in idxs]
        raw_quary_texts = [data[idx]["raw_quary_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        quary_durations = [data[idx]["quary_duration"] for idx in idxs]
        face_embs = np.stack([data[idx]["face_emb"] for idx in idxs])   # (B, 768)

        text_lens = np.array([text.shape[0] for text in texts])
        quary_text_lens = np.array([text.shape[0] for text in quary_texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        quary_texts = pad_1D(quary_texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        quary_durations = pad_1D(quary_durations)

        out = (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            raw_quary_texts,
            quary_texts,
            quary_text_lens,
            max(quary_text_lens),
            quary_durations,
            face_embs,
        )
        # Append ECAPA spk_embs (B, 192) as batch[18] when present.
        if "spk_emb" in data[idxs[0]]:
            spk_embs = np.stack([data[idx]["spk_emb"] for idx in idxs])
            out = out + (spk_embs,)
        return out

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class BatchInferenceDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        return (basename, speaker_id, phone, raw_text, mel, pitch, energy, duration)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        mels = [d[4] for d in data]
        pitches = [d[5] for d in data]
        energies = [d[6] for d in data]
        durations = [d[7] for d in data]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        ref_infos = list()
        for _, (m, p, e, d) in enumerate(zip(mels, pitches, energies, durations)):
            if self.pitch_feature_level == "phoneme_level":
                pitch = expand(p, d)
            else:
                pitch = p
            if self.energy_feature_level == "phoneme_level":
                energy = expand(e, d)
            else:
                energy = e
            ref_infos.append((m.T, pitch, energy))

        texts = pad_1D(texts)
        mels = pad_2D(mels)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            ref_infos,
        )

