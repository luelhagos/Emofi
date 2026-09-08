"""
Second, held-out intensity judge: a logistic-regression probe on mean-pooled
WavLM (microsoft/wavlm-base-plus, layer 9) features, fit on real MEAD training
audio only, then used to score real test audio (the probe's own ceiling) and
any number of synthesized test dirs. Reports per-level recall and UA.

Inputs:  preprocessed dir (train.txt / test.txt), real wav tree, real test
         wav dir, synth wav dirs (<basename>.wav)
Outputs: --out_txt report; features cached as .npz under --cache_dir

Usage:
    python3 scripts/eval_intensity_probe.py \
        --preprocessed data/MEAD \
        --raw_dir      data/MEAD_processed \
        --test_gt_dir  data/MEAD_processed \
        --synth_dirs   eval/test eval/test_other \
        --cache_dir    eval/intensity_probe_cache --out_txt eval/intensity_probe_report.txt
"""

import argparse
import os

import numpy as np
import torch

LV = ("level1", "level2", "level3")


def level_of(base):
    for t in base.split("_"):
        if t in LV:
            return LV.index(t)
    return None


def read_manifest(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 4:
                rows.append(parts[0])
    return rows


def index_raw(raw_dir):
    idx = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".wav"):
                idx[f[:-4]] = os.path.join(root, f)
    return idx


class Extractor:
    def __init__(self, model_name, layer, device):
        from transformers import AutoModel, AutoFeatureExtractor
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.layer = layer
        self.device = device

    @torch.no_grad()
    def __call__(self, wav_path):
        import librosa
        y, _ = librosa.load(wav_path, sr=16000)
        y = y[: 16000 * 10]                                   # cap 10 s
        inp = self.fe(y, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        out = self.model(inp, output_hidden_states=True)
        return out.hidden_states[self.layer].mean(dim=1).squeeze(0).cpu().numpy()


def featurize(bases, path_of, ex, cache):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"], list(z["kept"])
    X, y, kept = [], [], []
    for i, b in enumerate(bases):
        lv = level_of(b)
        p = path_of(b)
        if lv is None or p is None or not os.path.exists(p):
            continue
        X.append(ex(p)); y.append(lv); kept.append(b)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(bases)}")
    X, y = np.stack(X), np.array(y)
    np.savez(cache, X=X, y=y, kept=np.array(kept))
    return X, y, kept


def score(name, X, y, clf, out_lines):
    pred = clf.predict(X)
    rec = [float((pred[y == i] == i).mean()) for i in range(3)]
    ua = float(np.mean(rec))
    line = (f"{name:42} L1 {rec[0]:.3f}  L2 {rec[1]:.3f}  L3 {rec[2]:.3f} | UA {ua:.3f} "
            f"(n={len(y)})")
    print(line); out_lines.append(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed", required=True)
    p.add_argument("--raw_dir", required=True)
    p.add_argument("--test_gt_dir", required=True)
    p.add_argument("--synth_dirs", nargs="+", required=True)
    p.add_argument("--model", default="microsoft/wavlm-base-plus")
    p.add_argument("--layer", type=int, default=9)
    p.add_argument("--max_per_level", type=int, default=1500,
                   help="Cap real train utts per level (speed); 0 = all.")
    p.add_argument("--cache_dir", default="eval/intensity_probe_cache")
    p.add_argument("--out_txt", default="eval/intensity_probe_report.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    ex = Extractor(args.model, args.layer, args.device)

    print("[1/3] real train features (probe fit set)")
    train_bases = read_manifest(os.path.join(args.preprocessed, "train.txt"))
    if args.max_per_level:
        import random
        random.Random(0).shuffle(train_bases)
        per = {i: 0 for i in range(3)}; capped = []
        for b in train_bases:
            lv = level_of(b)
            if lv is not None and per[lv] < args.max_per_level:
                per[lv] += 1; capped.append(b)
        train_bases = capped
    raw_idx = index_raw(os.path.expandvars(args.raw_dir))
    Xtr, ytr, _ = featurize(train_bases, lambda b: raw_idx.get(b), ex,
                            os.path.join(args.cache_dir, "train.npz"))

    print("[2/3] fit probe (logistic regression, real audio only)")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    clf.fit(Xtr, ytr)

    print("[3/3] score real test (ceiling) + synth dirs")
    out_lines = [f"second intensity judge: {args.model} layer {args.layer}, "
                 f"LR probe fit on real train (n={len(ytr)})"]
    test_bases = read_manifest(os.path.join(args.preprocessed, "test.txt"))
    for name, d in [("real test (probe ceiling)", args.test_gt_dir)] + \
                   [(os.path.basename(d.rstrip("/")), d) for d in args.synth_dirs]:
        d = os.path.expandvars(d)
        X, y, _ = featurize(test_bases, lambda b: os.path.join(d, f"{b}.wav"), ex,
                            os.path.join(args.cache_dir,
                                         f"{os.path.basename(d.rstrip('/'))}.npz"))
        score(name, X, y, clf, out_lines)

    with open(args.out_txt, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\nreport -> {args.out_txt}")


if __name__ == "__main__":
    main()
