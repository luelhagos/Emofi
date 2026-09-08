"""
Erase the continuous arousal (or A/D/V) axis from the ECAPA speaker-embedding
cache with LEACE.

Steps: (1) score every cache stem's real wav with the audeering A/D/V model
(cached to --scores_json); (2) fit LEACE with Z = [arousal] or [A, D, V];
(3) report arousal R^2 / r from the embedding plus emotion and speaker probes
before/after; (4) optionally write the stripped cache (unit-norm <stem>.pt).
Can be composed with the categorical strip by passing the emotion-stripped
cache as input.

Usage:
    python3 scripts/leace_arousal_strip.py \
        --spk_emb_cache data/MEAD/spk_emb_cache_emostripped \
        --real_dir     data/MEAD_processed \
        --scores_json  eval/real_arousal_scores.json \
        --erased_out   data/MEAD/spk_emb_cache_emo_ar_stripped \
        --out_txt      eval/leace_arousal_strip_report.txt
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_ref_arousal_bleed import load_model, arousal_of          # audeering
from probe_ref_emotion_leak import (fit_leace, apply_leace,
                                    load_embeddings, probe)

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spk_emb_cache", required=True,
                   help="input ECAPA cache (pass the emotion-stripped one to "
                        "compose category + arousal erasure)")
    p.add_argument("--real_dir", required=True, help="real wavs <stem>.wav")
    p.add_argument("--scores_json", default="eval/real_arousal_scores.json",
                   help="audeering A/D/V per stem; computed once, reused")
    p.add_argument("--dims", choices=["arousal", "adv"], default="adv",
                   help="erase arousal only (1 dim) or arousal+dominance+valence (3)")
    p.add_argument("--erased_out", default=None)
    p.add_argument("--out_txt", default="leace_arousal_strip_report.txt")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for k in ("spk_emb_cache", "real_dir", "scores_json", "erased_out"):
        v = getattr(args, k)
        if v:
            setattr(args, k, os.path.expandvars(os.path.expanduser(v)))

    stems, X, spk, emo = load_embeddings(args.spk_emb_cache)
    print(f"[ar-strip] {len(stems)} embeddings from {args.spk_emb_cache}")

    # 1. audeering A/D/V for every stem's real wav (cached)
    scores = {}
    if os.path.exists(args.scores_json):
        scores = json.load(open(args.scores_json))
        print(f"[ar-strip] loaded {len(scores)} cached scores")
    missing = [s for s in stems if s not in scores]
    if missing:
        print(f"[ar-strip] scoring {len(missing)} real wavs with audeering ...")
        model, ext = load_model(args.device)

        # arousal_of returns arousal only; all 3 dims are needed here
        import librosa
        with torch.no_grad():
            for i, stem in enumerate(missing):
                wp = os.path.join(args.real_dir, f"{stem}.wav")
                if not os.path.exists(wp):
                    continue
                try:
                    y, _ = librosa.load(wp, sr=16000, mono=True)
                    xin = ext(y, sampling_rate=16000,
                              return_tensors="pt")["input_values"].to(args.device)
                    _, adv = model(xin)
                    scores[stem] = [float(adv[0, 0]), float(adv[0, 1]), float(adv[0, 2])]
                except Exception as e:
                    print(f"[err] {stem}: {e}")
                if (i + 1) % 500 == 0:
                    print(f"  {i+1}/{len(missing)}")
                    json.dump(scores, open(args.scores_json, "w"))  # checkpoint
        os.makedirs(os.path.dirname(args.scores_json) or ".", exist_ok=True)
        json.dump(scores, open(args.scores_json, "w"))
        print(f"[ar-strip] scores -> {args.scores_json}")

    keep = np.array([s in scores for s in stems])
    stems = [s for s, k in zip(stems, keep) if k]
    X, spk, emo = X[keep], spk[keep], emo[keep]
    Z = np.array([scores[s] for s in stems], dtype=np.float64)   # (n, 3) A/D/V
    if args.dims == "arousal":
        Z = Z[:, :1]
    print(f"[ar-strip] fitting LEACE on Z dims={Z.shape[1]} over n={len(stems)}")

    # 2. fit + apply
    mu, A = fit_leace(X, Z)
    Xe = apply_leace(X, mu, A)

    # 3. report
    spk_ids = {s: i for i, s in enumerate(sorted(set(spk)))}
    emo_ids = {e: i for i, e in enumerate(sorted(set(emo)))}
    y_spk = np.array([spk_ids[s] for s in spk])
    y_emo = np.array([emo_ids[e] for e in emo])
    idx_tr, idx_te = train_test_split(np.arange(len(stems)),
                                      test_size=args.test_frac,
                                      random_state=args.seed, stratify=y_spk)
    ar = Z[:, 0]

    def r2_and_r(Xm):
        m = Ridge(alpha=1.0).fit(Xm[idx_tr], ar[idx_tr])
        pred = m.predict(Xm[idx_te])
        ss = 1 - ((ar[idx_te] - pred) ** 2).sum() / ((ar[idx_te] - ar[idx_te].mean()) ** 2).sum()
        return ss, float(np.corrcoef(pred, ar[idx_te])[0, 1])

    L = [f"=== CONTINUOUS-AROUSAL LEACE: {args.spk_emb_cache} "
         f"(n={len(stems)}, erased dims={Z.shape[1]}) ==="]
    for tag, Xm in (("before", X), ("after", Xe)):
        ss, r = r2_and_r(Xm)
        L.append(f"\n{tag}:")
        L.append(f"  arousal from ECAPA (linear, want low after): R2={ss:.4f}  r={r:.4f}")
        probe("  emotion (linear/MLP)", Xm[idx_tr], y_emo[idx_tr],
              Xm[idx_te], y_emo[idx_te], 1 / len(emo_ids), L)
        probe("  speaker (want held)", Xm[idx_tr], y_spk[idx_tr],
              Xm[idx_te], y_spk[idx_te], 1 / len(spk_ids), L)
    L.append(f"\nmean |delta| per embedding: {np.abs(Xe - X).mean():.6f}")

    report = "\n".join(L)
    print("\n" + report)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")

    # 4. write erased cache
    if args.erased_out:
        os.makedirs(args.erased_out, exist_ok=True)
        for stem, v in zip(stems, Xe):
            t = torch.from_numpy(v.astype(np.float32))
            t = t / (t.norm() + 1e-9)          # ECAPA convention: unit norm
            torch.save(t, os.path.join(args.erased_out, f"{stem}.pt"))
        print(f"[ar-strip] wrote {len(stems)} arousal-stripped embeddings -> "
              f"{args.erased_out}")


if __name__ == "__main__":
    main()
