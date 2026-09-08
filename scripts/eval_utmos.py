"""
UTMOS (utmos22-strong) over a directory of wavs -- objective naturalness
prediction. Loads the checkpoint once via torch.hub (tarepan/SpeechMOS),
scores every wav, and reports the corpus mean +/- 95% CI plus per-emotion and
per-level breakdowns when filenames follow <spk>_<emotion>_<level>_<utt>.wav
(files with other names are scored and included in the corpus mean only).

Usage:
    python3 scripts/eval_utmos.py \
        --wav_dirs eval/test data/MEAD_processed \
        --out_tsv  eval/utmos_scores.tsv

Requires: torch, torchaudio (checkpoint downloads on first run).
"""

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dirs", nargs="+", required=True)
    p.add_argument("--out_tsv", default=None,
                   help="Per-file scores: dir<TAB>file<TAB>utmos")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                           trust_repo=True).to(args.device).eval()

    rows = []
    for d in args.wav_dirs:
        d = os.path.expandvars(os.path.expanduser(d))
        wavs = sorted(glob.glob(os.path.join(d, "*.wav")))
        scores, by_emo, by_lvl = [], defaultdict(list), defaultdict(list)
        for p in wavs:
            wave, sr = torchaudio.load(p)
            wave = wave.mean(0, keepdim=True).to(args.device)
            with torch.no_grad():
                s = float(model(wave, sr).squeeze())
            scores.append(s)
            rows.append((d, os.path.basename(p), s))
            parts = os.path.splitext(os.path.basename(p))[0].split("_")
            if len(parts) >= 3 and parts[1] and parts[2].startswith("level"):
                by_emo[parts[1]].append(s)
                by_lvl[parts[2]].append(s)
        a = np.array(scores)
        ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        print(f"\n=== {d}  (n={len(a)})")
        print(f"UTMOS mean {a.mean():.3f} +/- {ci:.3f}  median {np.median(a):.3f}")
        if by_emo:
            print("  by emotion: " + "  ".join(
                f"{e} {np.mean(v):.3f}" for e, v in sorted(by_emo.items())))
            print("  by level:   " + "  ".join(
                f"{l} {np.mean(v):.3f}" for l, v in sorted(by_lvl.items())))

    if args.out_tsv:
        out = os.path.expandvars(os.path.expanduser(args.out_tsv))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            for d, name, s in rows:
                f.write(f"{d}\t{name}\t{s:.4f}\n")
        print(f"\n[utmos] per-file scores -> {out}")


if __name__ == "__main__":
    main()
