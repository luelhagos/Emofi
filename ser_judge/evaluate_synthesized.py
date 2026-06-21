import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from dataset import (
    collate_fn_synthesized,
    EMOTIONS, LEVELS,
    NUM_EMOTION_CLASSES, NUM_LEVEL_CLASSES,
)
from model import EmotionClassifier, OrdinalLoss


class Tee:
    """Redirects print() to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.file     = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        # Newer transformers calls sys.stdout.isatty() in its loading report.
        return False

    def __getattr__(self, name):
        # Fall back to the real terminal for anything else (encoding, fileno, ...).
        return getattr(self.terminal, name)


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class WavDirDataset(Dataset):
    def __init__(self, wav_dir, target_sr=16000, max_duration=17.0):
        self.samples = []

        for fname in sorted(os.listdir(wav_dir)):
            if not fname.endswith(".wav"):
                continue

            parts = fname.replace(".wav", "").split("_")
            # accept speaker_emotion_level AND speaker_emotion_level_index (3+ parts);
            # emotion/level are always positions 1 and 2
            if len(parts) < 3:
                continue

            emotion, level = parts[1], parts[2]

            if emotion not in EMOTIONS or level not in LEVELS:
                continue

            self.samples.append({
                "path"         : os.path.join(wav_dir, fname),
                "emotion_label": EMOTIONS.index(emotion),
                "level_label"  : LEVELS.index(level),
                "filename"     : fname,
                "emotion"      : emotion,
                "level"        : level,
            })

        self.target_sr = target_sr
        self.max_len   = int(target_sr * max_duration)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        wav, _ = librosa.load(s["path"], sr=self.target_sr)

        if len(wav) > self.max_len:
            wav = wav[:self.max_len]
        else:
            wav = np.pad(wav, (0, self.max_len - len(wav)))

        return {
            "waveform"     : torch.tensor(wav, dtype=torch.float),
            "emotion_label": torch.tensor(s["emotion_label"]),
            "level_label"  : torch.tensor(s["level_label"]),
            "filename"     : s["filename"],
        }


# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────
def ordinal_logits_to_probs(logits):
    sig  = torch.sigmoid(logits)
    N, Km1 = sig.shape
    K    = Km1 + 1
    probs = []
    for k in range(K):
        if k == 0:
            probs.append(1 - sig[:, 0])
        elif k == K - 1:
            probs.append(sig[:, -1])
        else:
            probs.append(sig[:, k - 1] * (1 - sig[:, k]))
    return torch.stack(probs, dim=1)


def compute_ece(conf, pred, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.sum(mask) == 0:
            continue
        acc      = np.mean(pred[mask] == labels[mask])
        avg_conf = np.mean(conf[mask])
        ece     += np.abs(acc - avg_conf) * np.sum(mask) / len(conf)
    return ece


def plot_roc(probs, labels, class_names, save_path):
    plt.figure()
    for i, name in enumerate(class_names):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _   = roc_curve(binary_labels, probs[:, i])
        roc_auc       = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def print_confusion(matrix, labels, title):
    col_w = max(max(len(l) for l in labels), 5)
    print(f"\n{title}")
    print(" " * (col_w + 2) + "  ".join(f"{l:>{col_w}}" for l in labels))
    for i, row_label in enumerate(labels):
        row = "  ".join(f"{matrix[i, j]:>{col_w}}" for j in range(len(labels)))
        print(f"{row_label:<{col_w}}  {row}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav_dir",    required=True)
    parser.add_argument("--output_txt", default="per_file.txt")
    args = parser.parse_args()

    # ── redirect all prints to stdout + output_txt ────────────────────────────
    tee        = Tee(args.output_txt)
    sys.stdout = tee

    try:
        _run(args)
    finally:
        sys.stdout = tee.terminal
        tee.close()


def _run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt        = torch.load(args.checkpoint, map_location="cpu")
    saved_args  = argparse.Namespace(**ckpt["args"])
    use_ordinal = getattr(saved_args, "use_ordinal", False)

    print(f"[eval] checkpoint : {args.checkpoint}")
    print(f"[eval] wav_dir    : {args.wav_dir}")
    print(f"[eval] output_txt : {args.output_txt}")
    print(f"[eval] ordinal    : {use_ordinal}")

    model = EmotionClassifier(
        pretrained_name=saved_args.pretrained,
        num_emotions=NUM_EMOTION_CLASSES,
        num_levels=NUM_LEVEL_CLASSES,
        label_mode=saved_args.label_mode,
        num_joint=None,
        freeze_cnn=False,
        freeze_encoder_layers=0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds     = WavDirDataset(args.wav_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False,
                        collate_fn=collate_fn_synthesized)

    print(f"[eval] {len(ds)} samples found in {args.wav_dir}")

    all_e_probs, all_e_preds, all_e_labels = [], [], []
    all_l_probs, all_l_preds, all_l_labels = [], [], []
    per_file_lines = []

    for batch in loader:
        wav = batch["waveform"].to(device)

        e_logits, l_logits = model(wav)

        # emotion
        e_probs = F.softmax(e_logits, dim=-1)
        e_pred  = e_probs.argmax(dim=-1).cpu()
        e_conf  = e_probs.max(dim=-1).values.cpu()

        # intensity
        if use_ordinal:
            l_probs = ordinal_logits_to_probs(l_logits)
            l_pred  = OrdinalLoss.predict(l_logits).cpu()
        else:
            l_probs = F.softmax(l_logits, dim=-1)
            l_pred  = l_logits.argmax(dim=-1).cpu()

        l_conf  = l_probs.max(dim=-1).values.cpu()
        e_label = batch["emotion_label"]
        l_label = batch["level_label"]

        for i in range(len(e_pred)):
            fname = batch["filename"][i]
            per_file_lines.append(
                f"{fname}|"
                f"{int(e_pred[i] == e_label[i])}|{e_conf[i]:.4f}|"
                f"{int(l_pred[i] == l_label[i])}|{l_conf[i]:.4f}"
            )

        all_e_probs.append(e_probs.cpu())
        all_e_preds.append(e_pred)
        all_e_labels.append(e_label)
        all_l_probs.append(l_probs.cpu())
        all_l_preds.append(l_pred)
        all_l_labels.append(l_label)

    # ── concat ────────────────────────────────────────────────────────────────
    e_probs  = torch.cat(all_e_probs).numpy()
    e_preds  = torch.cat(all_e_preds).numpy()
    e_labels = torch.cat(all_e_labels).numpy()
    l_probs  = torch.cat(all_l_probs).numpy()
    l_preds  = torch.cat(all_l_preds).numpy()
    l_labels = torch.cat(all_l_labels).numpy()

    # ── emotion metrics ───────────────────────────────────────────────────────
    e_conf_mat = np.zeros((NUM_EMOTION_CLASSES, NUM_EMOTION_CLASSES), dtype=int)
    for gt, pr in zip(e_labels, e_preds):
        e_conf_mat[gt, pr] += 1

    per_class_recall = np.where(
        e_conf_mat.sum(axis=1) > 0,
        e_conf_mat.diagonal() / e_conf_mat.sum(axis=1),
        0.0,
    )
    UA = per_class_recall[e_conf_mat.sum(axis=1) > 0].mean()
    WA = (e_preds == e_labels).mean()

    print("\n" + "=" * 55)
    print("Emotion classification")
    print("=" * 55)
    print(f"UA: {UA:.4f}  WA: {WA:.4f}")
    print("\nPer-class recall:")
    for i, name in enumerate(EMOTIONS):
        total = e_conf_mat[i].sum()
        if total > 0:
            print(f"  {name:<12}  {per_class_recall[i]:.4f}  "
                  f"({e_conf_mat[i, i]}/{total})")
    print_confusion(e_conf_mat, EMOTIONS, "Emotion confusion matrix")

    # ── intensity metrics ─────────────────────────────────────────────────────
    l_conf_mat = np.zeros((NUM_LEVEL_CLASSES, NUM_LEVEL_CLASSES), dtype=int)
    for gt, pr in zip(l_labels, l_preds):
        l_conf_mat[gt, pr] += 1

    l_per_class = np.where(
        l_conf_mat.sum(axis=1) > 0,
        l_conf_mat.diagonal() / l_conf_mat.sum(axis=1),
        0.0,
    )
    l_UA = l_per_class[l_conf_mat.sum(axis=1) > 0].mean()
    l_WA = (l_preds == l_labels).mean()

    print("\n" + "=" * 55)
    print("Intensity classification")
    print("=" * 55)
    print(f"UA: {l_UA:.4f}  WA: {l_WA:.4f}")
    print("\nPer-class recall:")
    for i, name in enumerate(LEVELS):
        total = l_conf_mat[i].sum()
        if total > 0:
            print(f"  {name:<8}  {l_per_class[i]:.4f}  "
                  f"({l_conf_mat[i, i]}/{total})")
    print_confusion(l_conf_mat, LEVELS, "Intensity confusion matrix")

    # ── confidence analysis ───────────────────────────────────────────────────
    print("\nPer-class confidence (emotion):")
    for i, name in enumerate(EMOTIONS):
        mask = e_preds == i
        if np.sum(mask) > 0:
            print(f"  {name:<12}  {e_probs[mask, i].mean():.4f}")

    print("\nPer-class confidence (intensity):")
    for i, name in enumerate(LEVELS):
        mask = l_preds == i
        if np.sum(mask) > 0:
            print(f"  {name:<8}  {l_probs[mask, i].mean():.4f}")

    print(f"\nECE emotion   : {compute_ece(e_probs.max(1), e_preds, e_labels):.4f}")
    print(f"ECE intensity : {compute_ece(l_probs.max(1), l_preds, l_labels):.4f}")

    # ── ROC curves ────────────────────────────────────────────────────────────
    roc_dir       = os.path.dirname(args.output_txt) or "."
    emotion_roc   = os.path.join(roc_dir, "emotion_roc.png")
    intensity_roc = os.path.join(roc_dir, "intensity_roc.png")
    plot_roc(e_probs, e_labels, EMOTIONS, emotion_roc)
    plot_roc(l_probs, l_labels, LEVELS,   intensity_roc)
    print(f"\nSaved ROC curves ==> {emotion_roc}, {intensity_roc}")

    # ── top confusions ────────────────────────────────────────────────────────
    confusion = defaultdict(int)
    for gt, pr in zip(e_labels, e_preds):
        if gt != pr:
            confusion[(EMOTIONS[gt], EMOTIONS[pr])] += 1

    print("\nTop emotion confusions:")
    for k, v in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k[0]:<12} ==> {k[1]:<12}  {v}")

    # ── per-file results ──────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Per-file results  "
          "(filename | emotion_correct | emotion_conf | level_correct | level_conf)")
    print(f"{'─'*55}")
    for line in per_file_lines:
        print(line)

    print(f"\n[eval] done — all output saved to {args.output_txt}")


if __name__ == "__main__":
    main()
