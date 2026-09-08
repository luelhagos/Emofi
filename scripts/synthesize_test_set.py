"""
Test-set synthesis.

For every utterance in the manifest, load the cached face embedding, run the
face-conditioned model, vocode, and save the wav. Produces
<out_dir>/<basename>.wav for downstream scoring.

Usage:
    python3 scripts/synthesize_test_set.py \
        --restore_step 30000 \
        -p config/MEAD/preprocess.yaml \
        -m config/MEAD/model.yaml \
        -t config/MEAD/train.yaml \
        --manifest test.txt \
        --combine_mode concat_linear --speaker_emb \
        --out_dir eval/test
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, ".")
from text import text_to_sequence
from utils.model import get_model, get_vocoder
from utils.tools import pad_1D, pad_2D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--restore_step", type=int, required=True)
    p.add_argument("-p", "--preprocess_config", required=True)
    p.add_argument("-m", "--model_config",      required=True)
    p.add_argument("-t", "--train_config",      required=True)
    p.add_argument("--manifest", default="test.txt",
                   help="Which manifest under preprocessed_path to synthesize.")
    p.add_argument("--out_dir",  required=True,
                   help="Where to write <basename>.wav.")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--combine_mode", type=str, default=None,
                   choices=[None, "sum", "gated", "concat_linear"],
                   help="3-input models: must match the combine_mode the model was "
                        "trained with. For the cross-speaker swap test use "
                        "scripts/synthesize_swap_test.py.")
    p.add_argument("--speaker_emb", action="store_true",
                   help="Speaker channel = cached ECAPA emb (spk_emb_cache/). The "
                        "paired eval uses the utterance's own spk_emb. Must match how "
                        "the model was trained.")
    p.add_argument("--postfilter_refine", action="store_true",
                   help="Apply the intensity postfilter (mu -> mu + Delta) before "
                        "vocoding. Needs a checkpoint trained with "
                        "model config postfilter.enabled.")
    p.add_argument("--postfilter_levels", type=str, default=None,
                   help="Comma-separated levels the postfilter is applied to "
                        "(reference recipe: 'level2,level3' -- low-intensity "
                        "utterances bypass it, as added texture over-energizes calm "
                        "speech). Default None = all levels.")
    p.add_argument("--predict_level", action="store_true",
                   help="Predict the intensity level from the face (intensity aux "
                        "head) instead of reading it from the filename -- the "
                        "image-only condition, in which the face supplies both the "
                        "emotion and its grade. Default off = the filename (oracle) "
                        "level.")
    p.add_argument("--level_interp", type=float, default=None,
                   help="Continuous intensity control: a value t in [0, 2] mapped "
                        "onto the postfilter's learned level embeddings by linear "
                        "interpolation (0=level1, 1=level2, 2=level3). Applies the "
                        "postfilter to every utterance with the interpolated "
                        "embedding (the level gate is bypassed).")
    p.add_argument("--pf_ramp", action="store_true",
                   help="With --level_interp or --predict_level_expected, scale the "
                        "postfilter residual by min(1, t / 0.5) so t=0 is the exact "
                        "identity and the t<0.5 bypass becomes a linear fade.")
    p.add_argument("--predict_level_expected", action="store_true",
                   help="Continuous image-only intensity: t = expected level from the "
                        "intensity head's probabilities (sum_l p_l*l in [0, 2]) "
                        "interpolates the postfilter level embedding per utterance; "
                        "t < 0.5 bypasses the postfilter unless --pf_ramp is set. "
                        "Mutually exclusive with --predict_level and --level_interp.")
    return p.parse_args()


def expand_path_vars(cfg):
    for k, v in list(cfg.get("path", {}).items()):
        if isinstance(v, str):
            cfg["path"][k] = os.path.expandvars(os.path.expanduser(v))


def main():
    args = parse_args()
    device = torch.device(args.device)

    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config      = yaml.load(open(args.model_config),      Loader=yaml.FullLoader)
    train_config      = yaml.load(open(args.train_config),      Loader=yaml.FullLoader)
    expand_path_vars(preprocess_config)
    expand_path_vars(train_config)
    configs = (preprocess_config, model_config, train_config)

    args.out_dir = os.path.expandvars(os.path.expanduser(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)

    model   = get_model(args, configs, device, train=False)   # eval mode
    vocoder = get_vocoder(model_config, device)

    preprocessed_path = preprocess_config["path"]["preprocessed_path"]
    cleaners          = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sampling_rate     = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    face_emb_dir      = os.path.join(preprocessed_path, "face_emb_cache")
    spk_emb_dir       = os.path.join(preprocessed_path, "spk_emb_cache")

    rows = []
    with open(os.path.join(preprocessed_path, args.manifest)) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 4:
                continue
            basename, speaker, phones, _ = parts
            rows.append((basename, speaker, phones))

    print(f"Synthesizing {len(rows)} utterances from {args.manifest} -> {args.out_dir}")
    skipped_face = skipped_mel = errored = written = 0
    postfilter_levels = (set(args.postfilter_levels.split(","))
                         if args.postfilter_levels else None)
    LEVELS = ("level1", "level2", "level3")
    pred_log = []    # --predict_level: (filename level, predicted level)
    tval_log = []    # --predict_level_expected: (filename level, continuous t)

    if args.predict_level_expected and (args.predict_level
                                        or args.level_interp is not None):
        raise SystemExit("--predict_level_expected is mutually exclusive with "
                         "--predict_level and --level_interp.")

    def _refine_interp(mel_pred, style_vec, t, bypass_below=None):
        """Postfilter with the level embedding linearly interpolated at t in
        [0, 2] (0=level1, 1=level2, 2=level3). Without --pf_ramp, t below
        `bypass_below` returns the decoder mel unchanged; with --pf_ramp the
        residual is scaled by s = min(1, t / 0.5), so t=0 is the identity."""
        if not args.pf_ramp and bypass_below is not None and t < bypass_below:
            return mel_pred
        lo = min(int(t), 1)
        frac = t - lo
        w = torch.zeros(1, 3, device=device)
        w[0, lo] = 1.0 - frac
        w[0, lo + 1] = frac
        refined = model.postfilter_refine(mel_pred, style_vec, level_weights=w)
        if args.pf_ramp:
            s = min(1.0, t / 0.5)
            return mel_pred + s * (refined - mel_pred)
        return refined

    def _synth_one(seq, mel, face_embs, spk_embs=None, level=None, emotion=None):
        """Run a single forward (+ optional postfilter) + vocoder.
        Returns float32 wav in [-1, 1]."""
        text_lens   = np.array([len(seq)])
        texts       = torch.from_numpy(pad_1D([seq])).long().to(device)
        src_lens    = torch.from_numpy(text_lens).to(device)
        mel_lens_np = np.array([mel.shape[0]])
        mels        = torch.from_numpy(pad_2D([mel])).float().to(device)
        mel_lens    = torch.from_numpy(mel_lens_np).to(device)
        with torch.no_grad():
            output = model(
                None,
                texts, src_lens, int(text_lens[0]),
                mels, mel_lens, int(mel_lens_np[0]),
                None, None, None,
                face_embs=face_embs,
                spk_embs=spk_embs,
                combine_mode=args.combine_mode,
            )
            mel_pred = output[0]                       # (1, T, 80)

            if args.postfilter_refine:
                if getattr(model, "postfilter_g", None) is None:
                    raise SystemExit("--postfilter_refine needs a checkpoint with "
                                     "model config postfilter.enabled.")
                if args.predict_level and level is not None:
                    pred = int(model.compute_intensity_aux(face_embs)
                               .argmax(-1).item())
                    pred_log.append((level, LEVELS[pred]))
                    level = LEVELS[pred]
                style_vec = model.compute_style_vector(
                    face_embs, spk_embs=spk_embs, combine_mode=args.combine_mode)
                if args.predict_level_expected:
                    # Continuous image-only control: t = expected level under
                    # the intensity head's softmax; t < 0.5 bypasses.
                    probs = model.compute_intensity_aux(face_embs).softmax(-1).squeeze(0)
                    t = float((probs * torch.arange(
                        3, device=probs.device, dtype=probs.dtype)).sum())
                    if level is not None:
                        pred_log.append((level, LEVELS[int(round(t))]))
                        tval_log.append((level, t))
                    mel_pred = _refine_interp(mel_pred, style_vec, t, bypass_below=0.5)
                elif args.level_interp is not None:
                    # Continuous control: soft mixture over the level embeddings.
                    t = min(max(float(args.level_interp), 0.0), 2.0)
                    mel_pred = _refine_interp(mel_pred, style_vec, t)
                elif postfilter_levels is None or level in postfilter_levels:
                    lvl_idx = (torch.tensor([LEVELS.index(level)], device=device)
                               if level in LEVELS else None)
                    mel_pred = model.postfilter_refine(mel_pred, style_vec, lvl_idx)
                # else: bypass (low-intensity utterances keep the decoder mel)

            return vocoder(mel_pred.transpose(1, 2)).squeeze().cpu().numpy()

    for basename, speaker, phones in tqdm(rows):
        level = basename.split("_")[2] if len(basename.split("_")) > 2 else None
        out_wav_path = os.path.join(args.out_dir, f"{basename}.wav")
        if os.path.exists(out_wav_path):
            written += 1
            continue

        face_path = os.path.join(face_emb_dir, f"{basename}.pt")
        if not os.path.exists(face_path):
            skipped_face += 1; continue

        mel_path = os.path.join(preprocessed_path, "mel", f"{speaker}-mel-{basename}.npy")
        if not os.path.exists(mel_path):
            skipped_mel += 1; continue

        spk_embs = None
        if args.speaker_emb:
            spk_path = os.path.join(spk_emb_dir, f"{basename}.pt")
            if not os.path.exists(spk_path):
                skipped_face += 1; continue
            spk_embs = torch.load(spk_path, map_location="cpu").float().unsqueeze(0).to(device)

        try:
            face_embs = torch.load(face_path, map_location="cpu").float().unsqueeze(0).to(device)
            mel       = np.load(mel_path)
            seq       = np.array(text_to_sequence(phones, cleaners))
            emotion = basename.split("_")[1] if len(basename.split("_")) > 1 else None
            wav = _synth_one(seq, mel, face_embs, spk_embs=spk_embs, level=level,
                             emotion=emotion)
            wav = np.clip(wav, -1.0, 1.0)
            wav = (wav * 32767).astype(np.int16)
            wavfile.write(out_wav_path, sampling_rate, wav)
            written += 1
        except Exception as e:
            print(f"[err] {basename}: {e}")
            errored += 1

    print(
        f"\nDone. Wrote {written}, skipped {skipped_face} missing face emb, "
        f"{skipped_mel} missing ref mel, errored {errored}"
    )

    if pred_log:
        correct = sum(1 for g, p in pred_log if g == p)
        print(f"\n[face->level prediction] overall acc {correct / len(pred_log):.4f} "
              f"(n={len(pred_log)})")
        for gl in LEVELS:
            sub = [p for g, p in pred_log if g == gl]
            if not sub:
                continue
            rec = sum(1 for p in sub if p == gl) / len(sub)
            conf = {l: sum(1 for p in sub if p == l) for l in LEVELS}
            print(f"  {gl}: recall {rec:.3f} (n={len(sub)})  -> "
                  + " ".join(f"{l}:{conf[l]}" for l in LEVELS))

    if tval_log:
        print("[expected-level t] per filename level (mean +/- std):")
        for gl in LEVELS:
            ts = np.array([t for g, t in tval_log if g == gl])
            if len(ts):
                print(f"  {gl}: t {ts.mean():.3f} +/- {ts.std():.3f}  (n={len(ts)}; "
                      f"t<0.5: {(ts < 0.5).mean():.1%})")


if __name__ == "__main__":
    main()
