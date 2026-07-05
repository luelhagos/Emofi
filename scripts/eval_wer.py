"""
WER / CER of synthesized speech via Whisper ASR --> intelligibility metric.

Reports corpus WER + CER and a per-file breakdown.

Usage:
    python3 scripts/eval_wer.py \
        --wav_dir  eval/test \
        --manifest data/MEAD/test.txt \
        --model    large \
        --out_txt  eval/wer.txt

Run on several dirs and compare; real audio = floor.
Requires: pip install openai-whisper jiwer
"""

import argparse
import os

import whisper
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation, Strip, RemoveMultipleSpaces


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir",  required=True)
    p.add_argument("--manifest", required=True,
                   help="test.txt: basename|speaker|phones|raw_text.")
    p.add_argument("--model", default="large",
                   help="Whisper size: tiny/base/small/medium/large.")
    p.add_argument("--out_txt", default=None)
    p.add_argument("--language", default="en")
    p.add_argument("--max_files", type=int, default=0,
                   help="0 = all; otherwise cap (for a quick run).")
    return p.parse_args()


def main():
    args = parse_args()
    for k in ("wav_dir", "manifest"):
        setattr(args, k, os.path.expandvars(os.path.expanduser(getattr(args, k))))

    # basename -> reference transcript
    refs = {}
    with open(args.manifest) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 4:
                refs[parts[0]] = parts[3]
    print(f"[wer] {len(refs)} reference transcripts loaded")

    norm = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

    print(f"[wer] loading Whisper '{args.model}' ...")
    model = whisper.load_model(args.model)

    wavs = sorted(f for f in os.listdir(args.wav_dir) if f.endswith(".wav"))
    if args.max_files:
        wavs = wavs[:args.max_files]

    hyps, gts, lines = [], [], []
    skipped = 0
    for i, fn in enumerate(wavs):
        base = fn[:-4]
        if base not in refs:
            skipped += 1; continue
        path = os.path.join(args.wav_dir, fn)
        try:
            r = model.transcribe(path, language=args.language, fp16=False)
            hyp = norm(r["text"]); gt = norm(refs[base])
            if not gt:
                continue
            w = wer(gt, hyp)
            hyps.append(hyp); gts.append(gt)
            lines.append(f"{base}|WER={w:.3f}|REF={gt}|HYP={hyp}")
        except Exception as e:
            print(f"[err] {base}: {e}"); skipped += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(wavs)}")

    corpus_wer = wer(gts, hyps) if gts else float("nan")
    corpus_cer = cer(gts, hyps) if gts else float("nan")

    summary = (f"\n=== WER eval: {args.wav_dir} ===\n"
               f"files scored: {len(gts)}  (skipped {skipped})\n"
               f"corpus WER: {corpus_wer:.4f}\n"
               f"corpus CER: {corpus_cer:.4f}\n")
    print(summary)

    if args.out_txt:
        args.out_txt = os.path.expandvars(os.path.expanduser(args.out_txt))
        with open(args.out_txt, "w") as f:
            f.write(summary + "\n" + "\n".join(lines) + "\n")
        print(f"[wer] per-file + summary -> {args.out_txt}")


if __name__ == "__main__":
    main()

