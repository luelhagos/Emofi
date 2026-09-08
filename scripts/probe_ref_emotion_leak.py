"""
Emotion-leak probe on the cached ECAPA speaker embeddings.

Reads the 192-d speaker-embedding cache, reports the label floor
I(emotion; speaker), and probes emotion-from-embedding (linear + MLP; want
near chance) and speaker-from-embedding (sanity; want high). If emotion is
decodable (or --erased_out is given), LEACE-strips the emotion concept,
re-probes, and optionally writes the stripped cache (L2-normalized <stem>.pt).

Usage:
    python3 scripts/probe_ref_emotion_leak.py \
        --spk_emb_cache data/MEAD/spk_emb_cache \
        --out_txt       eval/ref_emotion_leak_report.txt \
        [--erased_out   data/MEAD/spk_emb_cache_emostripped]

Requires: numpy, torch, scikit-learn.
"""

import os
import re
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score

SPK_RE = re.compile(r"^([MW]\d{3})")
EMO_RE = re.compile(r"^[MW]\d{3}_([a-zA-Z]+)_")


# LEACE (closed-form); same operator as leace_disentangle.py
def fit_leace(X, Z, ridge=1e-6):
    mu = X.mean(0); Xc = X - mu; Zc = Z - Z.mean(0); n = X.shape[0]
    Sigma = (Xc.T @ Xc) / n
    Sxz = (Xc.T @ Zc) / n
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, ridge, None)
    Whalf = evecs @ np.diag(evals ** -0.5) @ evecs.T
    Whalf_inv = evecs @ np.diag(evals ** 0.5) @ evecs.T
    M = Whalf @ Sxz
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s.max() == 0:
        return mu, np.zeros((X.shape[1], X.shape[1]))
    rank = int((s > 1e-6 * s.max()).sum())
    Ur = U[:, :rank]; P = Ur @ Ur.T
    return mu, Whalf_inv @ P @ Whalf


def apply_leace(X, mu, A):
    return X - (X - mu) @ A.T


def onehot(idx, k):
    Z = np.zeros((len(idx), k)); Z[np.arange(len(idx)), idx] = 1.0; return Z


def load_embeddings(cache_dir):
    stems, embs, spk, emo = [], [], [], []
    for fn in sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt")):
        stem = fn[:-3]
        ms, me = SPK_RE.match(stem), EMO_RE.match(stem)
        if not (ms and me):
            continue
        v = torch.load(os.path.join(cache_dir, fn), map_location="cpu").float().view(-1).numpy()
        embs.append(v); stems.append(stem); spk.append(ms.group(1)); emo.append(me.group(1).lower())
    return stems, np.stack(embs).astype(np.float64), np.array(spk), np.array(emo)


def probe(name, Xtr, ytr, Xte, yte, chance, lines):
    lin = LogisticRegression(max_iter=2000, n_jobs=-1).fit(Xtr, ytr).score(Xte, yte)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                        early_stopping=True).fit(Xtr, ytr).score(Xte, yte)
    lines.append(f"  {name:26s} linear={lin:.4f}  nonlinear(MLP)={mlp:.4f}  (chance={chance:.4f})")
    return lin, mlp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spk_emb_cache", required=True, help="dir of 192-d ECAPA <stem>.pt")
    p.add_argument("--out_txt", default="ref_emotion_leak_report.txt")
    p.add_argument("--erased_out", default=None, help="write emotion-stripped cache here")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.spk_emb_cache = os.path.expandvars(os.path.expanduser(args.spk_emb_cache))

    stems, X, spk, emo = load_embeddings(args.spk_emb_cache)
    spk_vals, emo_vals = sorted(set(spk)), sorted(set(emo))
    spk_idx = np.array([spk_vals.index(s) for s in spk])
    emo_idx = np.array([emo_vals.index(e) for e in emo])
    n_spk, n_emo = len(spk_vals), len(emo_vals)

    L = [f"=== emotion-from-ECAPA leak probe: {args.spk_emb_cache} ===",
         f"N={len(X)}  dim={X.shape[1]}  speakers={n_spk}  emotions={n_emo}",
         f"label floor  normalized MI(emotion; speaker) = "
         f"{normalized_mutual_info_score(emo_idx, spk_idx):.4f}", ""]

    # split stratified by emotion so both probes are measurable
    tr, te = train_test_split(np.arange(len(X)), test_size=args.test_frac,
                              random_state=args.seed, stratify=emo_idx)

    L.append("Before (raw ECAPA):")
    el, en = probe("emotion leak (want low)", X[tr], emo_idx[tr], X[te], emo_idx[te], 1.0/n_emo, L)
    probe("speaker (sanity, want high)", X[tr], spk_idx[tr], X[te], spk_idx[te], 1.0/n_spk, L)

    L += ["", "Verdict (before stripping):",
          f"  emotion nonlinear {en:.3f} vs chance {1/n_emo:.3f} -> "
          + ("emotion-invariant (no strip needed)"
             if en < 1.5/n_emo else
             "emotion is decodable from the speaker channel -> strip it (--erased_out)")]

    Xe = None
    if args.erased_out or en >= 1.5/n_emo:
        mu, A = fit_leace(X[tr], onehot(emo_idx[tr], n_emo))
        Xe = apply_leace(X, mu, A)
        L += ["", "After LEACE (emotion erased from ECAPA):"]
        probe("emotion leak (want low)", Xe[tr], emo_idx[tr], Xe[te], emo_idx[te], 1.0/n_emo, L)
        probe("speaker (want held high)", Xe[tr], spk_idx[tr], Xe[te], spk_idx[te], 1.0/n_spk, L)

    report = "\n".join(L)
    print(report)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\n[probe2] report -> {args.out_txt}")

    if args.erased_out and Xe is not None:
        os.makedirs(args.erased_out, exist_ok=True)
        for stem, v in zip(stems, Xe):
            t = torch.tensor(v, dtype=torch.float32)
            t = t / (t.norm() + 1e-9)
            torch.save(t, os.path.join(args.erased_out, f"{stem}.pt"))
        print(f"[probe2] wrote {len(stems)} emotion-stripped embeddings -> {args.erased_out}")


if __name__ == "__main__":
    main()
