"""
Cache ECAPA-TDNN speaker embeddings for every MEAD utterance.

Runs speechbrain/spkrec-ecapa-voxceleb on each real 16 kHz wav listed in the
manifests and writes one L2-normalized (192,) tensor per utterance basename.
Existing files are skipped.

Produces:
    <out_dir>/<basename>.pt

Usage:
    python3 scripts/cache_speaker_embeddings.py \
        --manifest_dir data/MEAD \
        --real_dir     data/MEAD_processed \
        --out_dir      data/MEAD/spk_emb_cache

Requires: pip install speechbrain torchaudio
"""

import argparse
import os

import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", required=True,
                   help="Dir with train/val/test.txt (basenames in col 1).")
    p.add_argument("--real_dir", required=True,
                   help="Dir of real MEAD wavs (<basename>.wav).")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    for k in ("manifest_dir", "real_dir", "out_dir"):
        setattr(args, k, os.path.expandvars(os.path.expanduser(getattr(args, k))))
    os.makedirs(args.out_dir, exist_ok=True)

    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    device = "cuda:0" if args.device.startswith("cuda") else "cpu"
    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )

    # unique basenames across all splits
    bases = set()
    for split in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(args.manifest_dir, split)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                b = line.split("|")[0].strip()
                if b:
                    bases.add(b)
    bases = sorted(bases)
    print(f"[cache] {len(bases)} unique utterances")

    written = already = missing = errored = 0
    for b in tqdm(bases):
        out = os.path.join(args.out_dir, f"{b}.pt")
        if os.path.exists(out):
            already += 1; continue
        wav_path = os.path.join(args.real_dir, f"{b}.wav")
        if not os.path.exists(wav_path):
            missing += 1; continue
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                e = clf.encode_batch(wav).squeeze().cpu()   # (192,)
            e = e / (e.norm() + 1e-9)
            torch.save(e, out)
            written += 1
        except Exception as ex:
            print(f"[err] {b}: {ex}"); errored += 1

    print(f"\nDone. wrote {written}, already {already}, missing wav {missing}, "
          f"errored {errored}. dim = 192 (ECAPA-VoxCeleb).")


if __name__ == "__main__":
    main()
