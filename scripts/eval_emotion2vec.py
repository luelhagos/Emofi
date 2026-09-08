"""
Score synthesized audio with emotion2vec as a second,
independent emotion judge, plus an embedding-similarity metric.

Two outputs per synthesized directory:
  1. Categorical accuracy: emotion2vec+ predicts 9 emotion classes; predictions
     are masked to the 6 MEAD emotions and argmaxed, then scored against the
     label in the filename (<spk>_<emotion>_<level>_<idx>.wav). The same judge
     is run on the real test audio first, which gives this judge's own ceiling.
  2. Emotion-embedding similarity: cosine between each synthesized utterance's
     emotion2vec embedding and (a) the same utterance's real-audio embedding,
     (b) the real-audio centroid of its emotion vs the best other-emotion
     centroid (margin > 0 means the synth sits in the right emotion region).

Usage:
    python3 scripts/eval_emotion2vec.py \
        --real_dir   data/MEAD_processed \
        --manifest   data/MEAD/test.txt \
        --synth_dirs eval/test \
        --out_txt    eval/emotion2vec_report.txt
"""

import argparse
import os

import numpy as np

MEAD_EMOTIONS = ["angry", "fear", "happy", "neutral", "sad", "surprised"]
# emotion2vec+ class names -> MEAD
E2V_TO_MEAD = {
    "angry": "angry",
    "fearful": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprised": "surprised",
}


def load_model(name, hub, device):
    from funasr import AutoModel
    return AutoModel(model=name, hub=hub, device=device, disable_update=True)


def label_names(labels):
    """emotion2vec labels are '<chinese>/<english>' pairs; keep the part after '/'."""
    return [l.split("/")[-1].strip().lower() for l in labels]


def score_dir(model, wav_dir, bases, device, tag):
    """Classify + embed every <base>.wav in wav_dir. Returns dicts keyed by base."""
    import librosa
    preds, embs = {}, {}
    todo = [b for b in bases if os.path.exists(os.path.join(wav_dir, b + ".wav"))]
    print(f"[e2v] {tag}: scoring {len(todo)} files from {wav_dir}")
    for i, b in enumerate(todo):
        path = os.path.join(wav_dir, b + ".wav")
        try:
            y, _ = librosa.load(path, sr=16000, mono=True)
            res = model.generate(y, fs=16000, granularity="utterance",
                                 extract_embedding=True)
            r = res[0]
            names = label_names(r["labels"])
            scores = np.asarray(r["scores"], dtype=np.float64)
            # mask to the 6 MEAD-mappable classes, renormalize, argmax
            masked = {E2V_TO_MEAD[n]: s for n, s in zip(names, scores)
                      if n in E2V_TO_MEAD}
            preds[b] = max(masked, key=masked.get)
            embs[b] = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
        except Exception as e:
            print(f"[err] {b}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i+1}/{len(todo)}")
    return preds, embs


def accuracy(preds):
    """WA + per-class recall (UA) from filename labels."""
    per = {e: [0, 0] for e in MEAD_EMOTIONS}      # correct, total
    for b, p in preds.items():
        true = b.split("_")[1]
        if true not in per:
            continue
        per[true][1] += 1
        per[true][0] += int(p == true)
    total = sum(t for _, t in per.values())
    correct = sum(c for c, _ in per.values())
    recalls = {e: (c / t if t else float("nan")) for e, (c, t) in per.items()}
    wa = correct / max(total, 1)
    ua = float(np.nanmean(list(recalls.values())))
    return wa, ua, recalls


def cosine(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, help="real wavs <base>.wav")
    ap.add_argument("--manifest", required=True,
                    help="manifest defining which basenames to score (test.txt)")
    ap.add_argument("--synth_dirs", nargs="+", required=True)
    ap.add_argument("--model", default="iic/emotion2vec_plus_large",
                    help="emotion2vec checkpoint (iic/emotion2vec_plus_large, "
                         "iic/emotion2vec_plus_base, iic/emotion2vec_base_finetuned)")
    ap.add_argument("--hub", default="hf", choices=["hf", "ms"],
                    help="model hub: HuggingFace or ModelScope")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_files", type=int, default=0, help="0 = all")
    ap.add_argument("--out_txt", default="eval/emotion2vec_report.txt")
    args = ap.parse_args()
    for k in ("real_dir", "manifest"):
        setattr(args, k, os.path.expandvars(getattr(args, k)))
    args.synth_dirs = [os.path.expandvars(d) for d in args.synth_dirs]

    bases = []
    for line in open(args.manifest):
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4:
            bases.append(parts[0])
    if args.max_files:
        bases = bases[: args.max_files]
    print(f"[e2v] {len(bases)} test utterances; model {args.model}")

    model = load_model(args.model, args.hub, args.device)

    L = [f"=== emotion2vec judge: {args.model} (6-class-masked) ===",
         f"manifest: {args.manifest}  n={len(bases)}", ""]

    # 1) the judge's own ceiling on real audio + reference embeddings
    real_preds, real_embs = score_dir(model, args.real_dir, bases,
                                      args.device, "real")
    wa, ua, rec = accuracy(real_preds)
    L.append(f"REAL audio (judge ceiling):  WA {wa:.4f}  UA {ua:.4f}")
    L.append("  per-class: " + "  ".join(f"{e} {rec[e]:.3f}" for e in MEAD_EMOTIONS))
    L.append("")

    # real-audio emotion centroids for the margin metric
    cents = {}
    for e in MEAD_EMOTIONS:
        vecs = [v for b, v in real_embs.items() if b.split("_")[1] == e]
        if vecs:
            c = np.mean(vecs, axis=0)
            cents[e] = c / (np.linalg.norm(c) + 1e-9)

    # 2) each synthesized directory
    for d in args.synth_dirs:
        preds, embs = score_dir(model, d, bases, args.device, os.path.basename(d))
        wa, ua, rec = accuracy(preds)
        L.append(f"{d}:  WA {wa:.4f}  UA {ua:.4f}")
        L.append("  per-class: " + "  ".join(f"{e} {rec[e]:.3f}" for e in MEAD_EMOTIONS))

        paired, margins = [], []
        for b, v in embs.items():
            if b in real_embs:
                paired.append(cosine(v, real_embs[b]))
            e = b.split("_")[1]
            if e in cents:
                own = cosine(v, cents[e])
                other = max(cosine(v, c) for k, c in cents.items() if k != e)
                margins.append(own - other)
        if paired:
            L.append(f"  emotion-embedding similarity to paired real: "
                     f"mean {np.mean(paired):.4f} +/- {np.std(paired):.4f}")
        if margins:
            pos = float(np.mean(np.array(margins) > 0))
            L.append(f"  centroid margin (own-emotion minus best other): "
                     f"mean {np.mean(margins):+.4f}, positive for {pos:.1%} of files")
        L.append("")

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"[e2v] report -> {args.out_txt}")


if __name__ == "__main__":
    main()

