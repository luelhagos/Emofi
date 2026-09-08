"""
LEACE speaker erasure on cached face embeddings.

Closed-form linear concept erasure (LEACE, Belrose et al. 2023) with concept =
speaker identity, applied to the 768-d face-embedding cache. Reports the label
floor I(emotion; speaker), linear + MLP speaker and emotion probes before and
after erasure, and the mean relative perturbation. Optionally writes the erased
embeddings as a new cache (L2-normalized, same <stem>.pt layout).

Mode A reads a precomputed cache (--face_emb_cache). Mode B computes embeddings
from the released face encoder (--checkpoint + --filelist_dir).

Usage:
    python3 scripts/leace_disentangle.py \
        --face_emb_cache data/MEAD/face_emb_cache \
        --out_txt        eval/leace_report.txt \
        [--fit_manifest  data/MEAD/train.txt] \
        [--erased_out    data/MEAD/face_emb_cache_leace]

Requires: numpy, torch, scikit-learn.
"""

import os
import re
import sys
import argparse
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SPK_RE = re.compile(r"^([MW]\d{3})")
EMO_RE = re.compile(r"^[MW]\d{3}_([a-zA-Z]+)_")


# LEACE (closed-form), implemented directly to avoid a library dependency
def fit_leace(X, Z, ridge=1e-6):
    """X: (n,d) float64, Z: (n,k) float64 one-hot. Returns (mu, A) s.t. the
    eraser is r(x) = x - (x - mu) @ A.T, which zeroes Cov(r(X), Z)."""
    mu = X.mean(0)
    Xc = X - mu
    Zc = Z - Z.mean(0)
    n = X.shape[0]
    Sigma = (Xc.T @ Xc) / n                       # (d,d)
    Sxz = (Xc.T @ Zc) / n                          # (d,k)
    evals, evecs = np.linalg.eigh(Sigma)
    evals = np.clip(evals, ridge, None)
    Whalf = evecs @ np.diag(evals ** -0.5) @ evecs.T   # Sigma^{-1/2}
    Whalf_inv = evecs @ np.diag(evals ** 0.5) @ evecs.T  # Sigma^{1/2}
    M = Whalf @ Sxz                                # (d,k): whitened cross-cov
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.size == 0 or s.max() == 0:
        return mu, np.zeros((X.shape[1], X.shape[1]))
    rank = int((s > 1e-6 * s.max()).sum())
    Ur = U[:, :rank]
    P = Ur @ Ur.T                                  # proj onto colspace(M)
    A = Whalf_inv @ P @ Whalf                      # affine erasure operator
    return mu, A


def apply_leace(X, mu, A):
    return X - (X - mu) @ A.T


def compute_embeddings_from_encoder(checkpoint, filelist_dir, image_root, device):
    """Run the face encoder over the filelist faces. Embeddings are
    L2-normalized to match the cache."""
    from torchvision import transforms
    from PIL import Image
    from face_encoder.model import FaceEmotionEncoder

    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["model_config"]
    model = FaceEmotionEncoder(embedding_dim=cfg["embedding_dim"],
                               num_emotions=cfg["num_emotions"],
                               num_intensities=cfg["num_intensities"],
                               pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    rows = []
    for split in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(filelist_dir, split)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 6:
                    rows.append((parts[0], parts[1], parts[4].strip().lower()))
    print(f"[leace] {len(rows)} filelist rows; computing face embeddings ...")

    face_cache, stems, embs, spk, emo = {}, [], [], [], []
    with torch.no_grad():
        for i, (audio, face, emotion) in enumerate(rows):
            stem = os.path.splitext(os.path.basename(audio))[0]
            ms = SPK_RE.match(stem)
            if not ms:
                continue
            fpath = face if os.path.isabs(face) else (
                os.path.join(image_root, face) if image_root else face)
            if fpath not in face_cache:
                if not os.path.exists(fpath):
                    continue
                x = tf(Image.open(fpath).convert("RGB")).unsqueeze(0).to(device)
                e = model.get_emotion_embedding(x)[0]
                e = e / (e.norm() + 1e-9)
                face_cache[fpath] = e.float().cpu().numpy()
            embs.append(face_cache[fpath]); stems.append(stem)
            spk.append(ms.group(1)); emo.append(emotion)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(rows)}")
    X = np.stack(embs).astype(np.float64)
    return stems, X, np.array(spk), np.array(emo)


def load_embeddings(cache_dir):
    stems, embs, spk, emo = [], [], [], []
    files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
    for fn in files:
        stem = fn[:-3]
        ms, me = SPK_RE.match(stem), EMO_RE.match(stem)
        if not (ms and me):
            continue
        v = torch.load(os.path.join(cache_dir, fn), map_location="cpu").float().view(-1).numpy()
        embs.append(v); stems.append(stem); spk.append(ms.group(1)); emo.append(me.group(1).lower())
    X = np.stack(embs).astype(np.float64)
    return stems, X, np.array(spk), np.array(emo)


def onehot(idx, k):
    Z = np.zeros((len(idx), k), dtype=np.float64)
    Z[np.arange(len(idx)), idx] = 1.0
    return Z


def probe(name, Xtr, ytr, Xte, yte, chance, lines):
    lin = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1).fit(Xtr, ytr).score(Xte, yte)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, early_stopping=True
                        ).fit(Xtr, ytr).score(Xte, yte)
    lines.append(f"  {name:22s} linear={lin:.4f}  nonlinear(MLP)={mlp:.4f}  (chance={chance:.4f})")
    return lin, mlp


def main():
    p = argparse.ArgumentParser()
    # Mode A: read the precomputed cache. Mode B: compute from the encoder
    # (pass --checkpoint + --filelist_dir).
    p.add_argument("--face_emb_cache", default=None, help="Mode A: cache dir of <stem>.pt")
    p.add_argument("--checkpoint", default=None, help="Mode B: face encoder checkpoint (.pt)")
    p.add_argument("--filelist_dir", default="filelists", help="Mode B: dir with train/val/test.txt")
    p.add_argument("--image_root", default=None, help="Mode B: prefix for non-absolute face paths")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_txt", default="leace_report.txt")
    p.add_argument("--erased_out", default=None, help="write LEACE'd embeddings here")
    p.add_argument("--fit_manifest", default=None,
                   help="restrict the LEACE fit + probes to stems listed in this "
                        "manifest (e.g. data/MEAD/train.txt) so the eraser is "
                        "calibrated on the training distribution. The eraser is "
                        "still applied and written for every loaded embedding.")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.checkpoint:
        stems, X, spk, emo = compute_embeddings_from_encoder(
            args.checkpoint, args.filelist_dir, args.image_root, args.device)
    elif args.face_emb_cache:
        stems, X, spk, emo = load_embeddings(
            os.path.expandvars(os.path.expanduser(args.face_emb_cache)))
    else:
        p.error("provide either --face_emb_cache (Mode A) or --checkpoint (Mode B)")
    spk_vals = sorted(set(spk)); emo_vals = sorted(set(emo))
    spk_idx = np.array([spk_vals.index(s) for s in spk])
    emo_idx = np.array([emo_vals.index(e) for e in emo])
    n_spk, n_emo = len(spk_vals), len(emo_vals)

    L = [f"=== LEACE speaker-erasure test on {args.face_emb_cache} ===",
         f"N={len(X)}  dim={X.shape[1]}  speakers={n_spk}  emotions={n_emo}",
         f"label floor  normalized MI(emotion; speaker) = "
         f"{normalized_mutual_info_score(emo_idx, spk_idx):.4f}  "
         f"(~0 -> identity removable without sacrificing emotion)", ""]

    # restrict the fit universe to the manifest if requested; still applied to all
    if args.fit_manifest:
        keep = set()
        with open(args.fit_manifest, encoding="utf-8") as f:
            for line in f:
                keep.add(line.strip("\n").split("|")[0])
        fit_ids = np.array([i for i, s in enumerate(stems) if s in keep])
        L.append(f"FIT set: {len(fit_ids)}/{len(X)} embeddings from "
                 f"{os.path.basename(args.fit_manifest)}\n")
    else:
        fit_ids = np.arange(len(X))

    # within-speaker split (every fit speaker in train & test), stratified by speaker
    tr, te = train_test_split(fit_ids, test_size=args.test_frac,
                              random_state=args.seed, stratify=spk_idx[fit_ids])

    # fit LEACE on the train embeddings, concept = speaker one-hot
    mu, A = fit_leace(X[tr], onehot(spk_idx[tr], n_spk))
    Xe = apply_leace(X, mu, A)
    drift = float(np.linalg.norm(Xe - X, axis=1).mean() / (np.linalg.norm(X, axis=1).mean() + 1e-9))
    L.append(f"mean relative perturbation from erasure: {drift:.4f}\n")

    L.append("Before erasure:")
    probe("speaker leakage", X[tr], spk_idx[tr], X[te], spk_idx[te], 1.0 / n_spk, L)
    probe("emotion (keep)", X[tr], emo_idx[tr], X[te], emo_idx[te], 1.0 / n_emo, L)
    L.append("\nAFTER LEACE erasure:")
    sl, sn = probe("speaker leakage", Xe[tr], spk_idx[tr], Xe[te], spk_idx[te], 1.0 / n_spk, L)
    el, en = probe("emotion (keep)", Xe[tr], emo_idx[tr], Xe[te], emo_idx[te], 1.0 / n_emo, L)

    L += ["", "Verdict:",
          f"  speaker linear {sl:.3f} / nonlinear {sn:.3f} vs chance {1/n_spk:.3f} "
          f"-> {'clean' if sn < 2/n_spk else 'nonlinear residual remains'}",
          f"  emotion nonlinear {en:.3f} vs chance {1/n_emo:.3f} "
          f"-> {'preserved' if en > 0.5 else 'degraded'}"]

    report = "\n".join(L)
    print(report)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\n[leace] report -> {args.out_txt}")

    if args.erased_out:
        os.makedirs(args.erased_out, exist_ok=True)
        for stem, v in zip(stems, Xe):
            t = torch.tensor(v, dtype=torch.float32)
            t = t / (t.norm() + 1e-9)          # keep L2-normalized like the cache
            torch.save(t, os.path.join(args.erased_out, f"{stem}.pt"))
        print(f"[leace] wrote {len(stems)} erased embeddings -> {args.erased_out}")


if __name__ == "__main__":
    main()
