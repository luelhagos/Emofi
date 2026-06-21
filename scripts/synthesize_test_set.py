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

sys.path.insert(0, ".")  # so `import text`, `import utils.*` resolve from repo root
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

    def _synth_one(seq, mel, face_embs, spk_embs=None):
        """Run a single forward + vocoder. Returns float32 wav in [-1, 1]."""
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
            return vocoder(mel_pred.transpose(1, 2)).squeeze().cpu().numpy()

    for basename, speaker, phones in tqdm(rows):
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
            wav = _synth_one(seq, mel, face_embs, spk_embs=spk_embs)
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


if __name__ == "__main__":
    main()
