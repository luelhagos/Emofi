"""
Continuous arousal / valence / dominance of speech with the pretrained
dimensional model audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
(inference only).

For each wav dir: mean +/- std per dimension, the arousal distribution
(min/p5/p50/p95/max), and per-intensity-level and per-emotion arousal when
basenames parse as <spk>_<emotion>_<level>_<utt>. With several dirs, also a
comparison table and paired CCC / Pearson of per-utterance arousal against a
reference dir (the one whose name contains 'real', else the first).

Usage (single dir):
    python3 scripts/eval_arousal.py \
        --wav_dir eval/test --out_txt eval/arousal.txt

Usage (comparison; utterances matched by basename):
    python3 scripts/eval_arousal.py \
        --wav_dir data/MEAD_processed eval/test eval/test_other \
        --out_txt eval/arousal_compare.txt

Requires: pip install transformers torch librosa numpy
"""

import argparse
import os
from collections import defaultdict

import json

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SR = 16000
# the audeering head outputs three values in this order
DIMS = ["arousal", "dominance", "valence"]


class RegressionHead(nn.Module):
    """Regression head from the audeering model card."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        # post_init() is required by current transformers (replaces init_weights())
        self.post_init()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        hidden = torch.mean(hidden, dim=1)
        return hidden, self.classifier(hidden)


def load_model_and_extractor(model_name, device):
    """Load the audeering model and feature extractor.

    The model's config.json has vocab_size=null (no token head); newer
    huggingface_hub versions reject that, so vocab_size is forced to a dummy
    int and only the feature extractor (not the full Processor) is loaded.
    """
    try:
        config = Wav2Vec2Config.from_pretrained(model_name)
        if getattr(config, "vocab_size", None) is None:
            config.vocab_size = 32
    except Exception:
        # strict validation failed inside from_pretrained: build from the raw json
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(model_name, "config.json")) as f:
            cfg = json.load(f)
        cfg["vocab_size"] = cfg.get("vocab_size") or 32
        config = Wav2Vec2Config(**cfg)

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name, config=config).to(device).eval()
    return model, extractor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wav_dir", required=True, nargs="+",
                   help="One or more wav dirs. Multiple -> comparison table.")
    p.add_argument("--out_txt", default=None)
    p.add_argument("--model_name", default=MODEL_NAME)
    p.add_argument("--max_files", type=int, default=0,
                   help="0 = all; otherwise cap per dir (quick smoke run).")
    p.add_argument("--device", default=None,
                   help="cuda / cpu. Default: cuda if available.")
    return p.parse_args()


def emotion_level_from_name(base):
    """Best-effort (emotion, level) from speaker_emotion_level_uttid names."""
    parts = base.split("_")
    emo = lvl = None
    for tok in parts:
        t = tok.lower()
        if t in ("angry", "anger", "happy", "sad", "surprised", "surprise",
                 "fear", "fearful", "neutral", "disgust", "contempt"):
            emo = t
        if t.startswith("level") or t in ("level1", "level2", "level3"):
            lvl = t
    return emo, lvl


@torch.no_grad()
def score_dir(wav_dir, model, processor, device, max_files):
    """Return (per_file list of (base, a, d, v), np.array of [a,d,v])."""
    wavs = sorted(f for f in os.listdir(wav_dir) if f.endswith(".wav"))
    if max_files:
        wavs = wavs[:max_files]

    rows, vecs = [], []
    for i, fn in enumerate(wavs):
        path = os.path.join(wav_dir, fn)
        try:
            y, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            if y.size == 0:
                continue
            inp = processor(y, sampling_rate=TARGET_SR, return_tensors="pt")
            x = inp["input_values"].to(device)
            _, logits = model(x)
            a, d, v = logits[0].cpu().numpy().tolist()
            rows.append((fn[:-4], a, d, v))
            vecs.append([a, d, v])
        except Exception as e:
            print(f"[err] {fn}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {os.path.basename(wav_dir)}: {i+1}/{len(wavs)}")
    return rows, (np.array(vecs) if vecs else np.empty((0, 3)))


def ccc(x, y):
    """Lin's Concordance Correlation Coefficient:
    2*cov(x,y) / (var(x) + var(y) + (mean(x)-mean(y))^2)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).mean()
    return 2 * cov / (x.var() + y.var() + (mx - my) ** 2 + 1e-12)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx, cy = x - x.mean(), y - y.mean()
    denom = np.sqrt((cx ** 2).sum() * (cy ** 2).sum()) + 1e-12
    return float((cx * cy).sum() / denom)


def breakdown(rows, key_idx):
    """Arousal values grouped by emotion and by level (parsed from names)."""
    by_emo, by_lvl = defaultdict(list), defaultdict(list)
    for base, a, d, v in rows:
        emo, lvl = emotion_level_from_name(base)
        if emo:
            by_emo[emo].append(a)
        if lvl:
            by_lvl[lvl].append(a)
    return by_emo, by_lvl


def main():
    args = parse_args()
    args.wav_dir = [os.path.expandvars(os.path.expanduser(d)) for d in args.wav_dir]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[arousal] loading {args.model_name} on {device} ...")
    model, processor = load_model_and_extractor(args.model_name, device)

    out_blocks = []
    table = []  # (label, n, mean_a, std_a, mean_v, mean_d)
    arousal_by_dir = {}  # label -> {basename: arousal} for paired CCC
    labels_in_order = []
    for wav_dir in args.wav_dir:
        if not os.path.isdir(wav_dir):
            print(f"[skip] not a dir: {wav_dir}")
            continue
        label = os.path.basename(wav_dir.rstrip("/"))
        print(f"[arousal] scoring {label} ...")
        rows, vecs = score_dir(wav_dir, model, processor, device, args.max_files)
        arousal_by_dir[label] = {base: a for base, a, d, v in rows}
        labels_in_order.append(label)
        if len(vecs) == 0:
            print(f"[warn] no scored files in {label}")
            continue
        m = vecs.mean(axis=0)
        s = vecs.std(axis=0)
        table.append((label, len(vecs), m[0], s[0], m[2], m[1]))

        by_emo, by_lvl = breakdown(rows, None)
        block = [f"=== {label}  (n={len(vecs)}) ===",
                 f"  arousal  : {m[0]:.4f} +/- {s[0]:.4f}",
                 f"  valence  : {m[2]:.4f} +/- {s[2]:.4f}",
                 f"  dominance: {m[1]:.4f} +/- {s[1]:.4f}"]
        # upper tail of the arousal distribution (probe headroom check)
        a_vals = vecs[:, 0]
        block.append("  arousal distribution: "
                     f"min {a_vals.min():.3f} | p5 {np.percentile(a_vals,5):.3f} | "
                     f"p50 {np.percentile(a_vals,50):.3f} | p95 {np.percentile(a_vals,95):.3f} "
                     f"| max {a_vals.max():.3f}")
        if by_lvl:
            # per-level mean +/- within-group std, and the L3-L1 conditional range
            block.append("  arousal by intensity level (mean +/- within-group std):")
            lvl_means = {}
            for lvl in sorted(by_lvl):
                vals = np.array(by_lvl[lvl])
                lvl_means[lvl] = vals.mean()
                block.append(f"    {lvl:8s}: {vals.mean():.4f} +/- {vals.std():.4f}  (n={len(vals)})")
            if "level1" in lvl_means and "level3" in lvl_means:
                block.append(f"    >> conditional range L3-L1: {lvl_means['level3']-lvl_means['level1']:+.4f}")
        if by_emo:
            block.append("  arousal by emotion (mean +/- within-group std):")
            for emo in sorted(by_emo, key=lambda e: -np.mean(by_emo[e])):
                vals = np.array(by_emo[emo])
                block.append(f"    {emo:10s}: {vals.mean():.4f} +/- {vals.std():.4f}  (n={len(vals)})")
        out_blocks.append("\n".join(block))
        print(out_blocks[-1] + "\n")

    # comparison table across dirs
    if table:
        hdr = f"\n{'dir':28s} {'n':>5s} {'arousal':>9s} {'std':>7s} {'valence':>9s} {'dominance':>10s}"
        sep = "-" * len(hdr)
        lines = [hdr, sep]
        for label, n, ma, sa, mv, md in table:
            lines.append(f"{label:28s} {n:5d} {ma:9.4f} {sa:7.4f} {mv:9.4f} {md:10.4f}")
        comp = "\n".join(lines)
        print(comp)
        out_blocks.append(comp)

    # paired CCC vs a reference dir (name containing 'real', else the first);
    # utterances matched by basename. Skipped for a single dir.
    if len(labels_in_order) > 1:
        ref = next((l for l in labels_in_order if "real" in l.lower()),
                   labels_in_order[0])
        ref_map = arousal_by_dir.get(ref, {})
        lines = [f"\n=== paired tracking vs reference '{ref}' ===",
                 f"{'dir':28s} {'matched':>8s} {'CCC':>8s} {'Pearson':>8s}"]
        lines.append("-" * len(lines[-1]))
        for label in labels_in_order:
            if label == ref:
                continue
            cur = arousal_by_dir.get(label, {})
            common = [b for b in cur if b in ref_map]
            if len(common) < 3:
                lines.append(f"{label:28s} {len(common):8d}  (too few matches)")
                continue
            xs = [ref_map[b] for b in common]
            ys = [cur[b] for b in common]
            lines.append(f"{label:28s} {len(common):8d} {ccc(xs, ys):8.4f} "
                         f"{pearson(xs, ys):8.4f}")
        cccblock = "\n".join(lines)
        print(cccblock)
        out_blocks.append(cccblock)

    if args.out_txt:
        args.out_txt = os.path.expandvars(os.path.expanduser(args.out_txt))
        with open(args.out_txt, "w") as f:
            f.write("\n\n".join(out_blocks) + "\n")
        print(f"\n[arousal] summary -> {args.out_txt}")


if __name__ == "__main__":
    main()
