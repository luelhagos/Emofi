"""
Reference-arousal bleed check on swap-test outputs.

Scores arousal (audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) for
(a) each swap output, (b) its reference utterance's real audio, (c) its face /
source utterance's real audio. Within each face-(emotion, level) group, reports
the correlation and regression slope of output arousal on reference arousal,
and the correlation on face-source arousal, plus size-weighted means.

Inputs:  swap wav dir, swap pairs file (base|face=<emo>|ref=<refbase>|ref_emo=<emo>),
         real wav dir
Output:  --out_txt report

Usage:
    python3 scripts/check_ref_arousal_bleed.py \
        --swap_dir eval/swap --pairs eval/swap_pairs.txt \
        --real_dir data/MEAD_processed \
        --out_txt eval/ref_arousal_bleed.txt
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model, Wav2Vec2PreTrainedModel,
)

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SR = 16000


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.post_init()

    def forward(self, input_values):
        h = torch.mean(self.wav2vec2(input_values)[0], dim=1)
        return h, self.classifier(h)


def load_model(device):
    """Load the audeering model; vocab_size=null in its config is patched to a dummy int."""
    try:
        config = Wav2Vec2Config.from_pretrained(MODEL_NAME)
        if getattr(config, "vocab_size", None) is None:
            config.vocab_size = 32
    except Exception:
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(MODEL_NAME, "config.json")) as f:
            cfg = json.load(f)
        cfg["vocab_size"] = cfg.get("vocab_size") or 32
        config = Wav2Vec2Config(**cfg)
    ext = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = EmotionModel.from_pretrained(MODEL_NAME, config=config).to(device).eval()
    return model, ext


@torch.no_grad()
def arousal_of(paths, model, ext, device, tag):
    """Return {path: arousal} for the given wav paths."""
    out = {}
    for i, p in enumerate(paths):
        try:
            y, _ = librosa.load(p, sr=SR, mono=True)
            if y.size == 0:
                continue
            x = ext(y, sampling_rate=SR, return_tensors="pt")["input_values"].to(device)
            _, adv = model(x)
            out[p] = float(adv[0, 0])          # arousal
        except Exception as e:
            print(f"[err] {p}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i+1}/{len(paths)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap_dir", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--out_txt", default="ref_arousal_bleed.txt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    for k in ("swap_dir", "pairs", "real_dir"):
        setattr(args, k, os.path.expandvars(getattr(args, k)))

    # pairs: base|face=<emo>|ref=<refbase>|ref_emo=<emo>
    pairs = {}
    for line in open(args.pairs):
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        base = parts[0]
        ref_base = parts[2].split("=", 1)[1]
        pairs[base] = ref_base
    print(f"[bleed] {len(pairs)} swap pairs")

    model, ext = load_model(args.device)

    out_paths = [os.path.join(args.swap_dir, f"{b}.wav") for b in pairs]
    out_paths = [p for p in out_paths if os.path.exists(p)]
    ref_paths = sorted({os.path.join(args.real_dir, f"{r}.wav") for r in pairs.values()})
    ref_paths = [p for p in ref_paths if os.path.exists(p)]
    face_paths = sorted({os.path.join(args.real_dir, f"{b}.wav") for b in pairs})
    face_paths = [p for p in face_paths if os.path.exists(p)]

    print(f"[bleed] scoring {len(out_paths)} outputs, {len(ref_paths)} ref reals, "
          f"{len(face_paths)} face reals ...")
    a_out = arousal_of(out_paths, model, ext, args.device, "outputs")
    a_ref = arousal_of(ref_paths, model, ext, args.device, "refs")
    a_face = arousal_of(face_paths, model, ext, args.device, "faces")

    # rows grouped by the face conditioning (emotion, level)
    grp = defaultdict(list)
    for base, ref_base in pairs.items():
        po = os.path.join(args.swap_dir, f"{base}.wav")
        pr = os.path.join(args.real_dir, f"{ref_base}.wav")
        pf = os.path.join(args.real_dir, f"{base}.wav")
        if po in a_out and pr in a_ref and pf in a_face:
            m = re.match(r"[MW]\d{3}_([A-Za-z]+)_(level\d+)_", base)
            if m:
                grp[(m.group(1), m.group(2))].append(
                    (a_out[po], a_ref[pr], a_face[pf]))

    def corr(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])

    def slope(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if len(x) < 3 or x.std() == 0:
            return np.nan
        return float(np.cov(x, y)[0, 1] / x.var())

    L = ["=== REF-AROUSAL BLEED CHECK (within face-(emotion,level) groups) ===",
         f"groups: {len(grp)}   total rows: {sum(len(v) for v in grp.values())}",
         "",
         f"{'group':22s} {'n':>4s} {'corr(out,REF)':>14s} {'slope_b':>8s} {'corr(out,FACE)':>15s}"]
    ref_corrs, ref_slopes, face_corrs, weights = [], [], [], []
    for (emo, lvl), rows in sorted(grp.items()):
        if len(rows) < 10:
            continue
        o = [r[0] for r in rows]; rr = [r[1] for r in rows]; ff = [r[2] for r in rows]
        cr, b, cf = corr(rr, o), slope(rr, o), corr(ff, o)
        L.append(f"{emo+'_'+lvl:22s} {len(rows):4d} {cr:14.3f} {b:8.3f} {cf:15.3f}")
        if not np.isnan(cr):
            ref_corrs.append(cr); ref_slopes.append(b); face_corrs.append(cf)
            weights.append(len(rows))
    w = np.array(weights, dtype=float); w /= w.sum()
    mean_cr = float(np.dot(w, ref_corrs)); mean_b = float(np.dot(w, ref_slopes))
    mean_cf = float(np.dot(w, [c if not np.isnan(c) else 0 for c in face_corrs]))
    L += ["",
          f"weighted mean within-group corr(out, REF arousal):  {mean_cr:+.3f}   slope b: {mean_b:+.3f}",
          f"weighted mean within-group corr(out, FACE arousal): {mean_cf:+.3f}",
          "",
          "Reading: |corr(out,REF)| < 0.1 -> no reference-intensity bleed;",
          "corr substantially > 0 with slope b > ~0.1 -> the reference's arousal",
          f"partially carries into the output prosody (r={mean_cr:.2f}, b={mean_b:.2f})."]
    report = "\n".join(L)
    print("\n" + report)
    with open(args.out_txt, "w") as f:
        f.write(report + "\n")
    print(f"\n[bleed] report -> {args.out_txt}")


if __name__ == "__main__":
    main()
