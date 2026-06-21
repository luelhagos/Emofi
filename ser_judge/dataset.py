"""
MEAD emotion dataset loader

Filelist format (pipe-separated):
  audio_path|image_path|text|phonemes|emotion|level

Emotions : angry, fear, happy, surprised, sad, neutral
Levels   : level1, level2, level3  (neutral only ever has level1)

Audio is stored at 22050 Hz; this loader resamples to 16000 Hz on-the-fly.
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ── Label maps ────────────────────────────────────────────────────────────────

EMOTIONS = ["angry", "fear", "happy", "surprised", "sad", "neutral"]
LEVELS   = ["level1", "level2", "level3"]

EMOTION2IDX = {e: i for i, e in enumerate(EMOTIONS)}
LEVEL2IDX   = {l: i for i, l in enumerate(LEVELS)}

# Joint label: (emotion, level) ==> single int
# neutral is always level1 ==> joint index stays consistent
def make_joint_map():
    idx = 0
    mapping = {}
    for e in EMOTIONS:
        if e == "neutral":
            mapping[(e, "level1")] = idx
            idx += 1
        else:
            for l in LEVELS:
                mapping[(e, l)] = idx
                idx += 1
    return mapping

JOINT2IDX = make_joint_map()
NUM_JOINT_CLASSES   = len(JOINT2IDX)   # 5*3 + 1 = 16
NUM_EMOTION_CLASSES = len(EMOTIONS)    # 6
NUM_LEVEL_CLASSES   = len(LEVELS)      # 3

print(f"[dataset] joint classes: {NUM_JOINT_CLASSES}, "
      f"emotion classes: {NUM_EMOTION_CLASSES}, "
      f"intensity classes: {NUM_LEVEL_CLASSES}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class MEADDataset(Dataset):
    """
    Loads audio from a MEAD-format filelist and resamples on-the-fly.

    Args:
        filelist_path : path to the pipe-separated filelist
        orig_sr       : sample rate of the stored .wav files (default 22050)
        target_sr     : sample rate wav2vec2 expects (default 16000)
        max_duration  : clip/pad audio to this many seconds (default 6)
        augment       : apply random augmentations (train only)
        label_mode    : "joint" | "two_head"
    """

    def __init__(
        self,
        filelist_path: str,
        orig_sr: int = 22050,
        target_sr: int = 16000,
        max_duration: float = 6.0,
        augment: bool = False,
        label_mode: str = "two_head",
    ):
        self.orig_sr      = orig_sr
        self.target_sr    = target_sr
        self.max_samples  = int(max_duration * target_sr)
        self.augment      = augment
        self.label_mode   = label_mode

        # resampler is stateless for fixed ratio, safe to share
        self.resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)

        # augmentation transforms (applied after resampling)
        self.noise_level = 0.005
        self.time_mask   = T.TimeMasking(time_mask_param=int(0.1 * target_sr))
        self.freq_mask   = None  # wav2vec2 takes raw waveform, not spectrogram

        self.samples = self._load_filelist(filelist_path)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _load_filelist(self, path: str):
        samples = []
        skipped = 0
        with open(path, "r") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 6:
                    print(f"[dataset] WARNING line {lineno}: expected 6 fields, got {len(parts)} — skipping")
                    skipped += 1
                    continue

                audio_path = parts[0]
                emotion    = parts[4].strip().lower()
                level      = parts[5].strip().lower()

                # sanity checks
                if emotion not in EMOTION2IDX:
                    print(f"[dataset] WARNING line {lineno}: unknown emotion '{emotion}' --> skipping")
                    skipped += 1
                    continue
                if level not in LEVEL2IDX:
                    print(f"[dataset] WARNING line {lineno}: unknown level '{level}' --> skipping")
                    skipped += 1
                    continue
                if emotion == "neutral" and level != "level1":
                    # force neutral to level1 regardless of what filelist says
                    level = "level1"

                samples.append({
                    "audio_path"  : audio_path,
                    "emotion"     : emotion,
                    "level"       : level,
                    "emotion_idx" : EMOTION2IDX[emotion],
                    "level_idx"   : LEVEL2IDX[level],
                    "joint_idx"   : JOINT2IDX[(emotion, level)],
                })

        print(f"[dataset] loaded {len(samples)} samples from {path}  "
              f"(skipped {skipped})")
        return samples

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load, resample to target_sr, mix to mono, return 1-D tensor."""
        waveform, sr = torchaudio.load(path)

        # mix down to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed (handles orig_sr ==> target_sr)
        if sr != self.target_sr:
            if sr == self.orig_sr:
                waveform = self.resampler(waveform)
            else:
                # fallback for files that differ from expected orig_sr
                waveform = T.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        return waveform.squeeze(0)  # (T,)

    def _pad_or_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        n = waveform.shape[0]
        if n >= self.max_samples:
            return waveform[: self.max_samples]
        pad = self.max_samples - n
        return torch.nn.functional.pad(waveform, (0, pad))

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        # additive Gaussian noise
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(waveform) * self.noise_level
            waveform = waveform + noise

        # random gain ±3 dB
        if torch.rand(1).item() < 0.5:
            gain = 10 ** (torch.empty(1).uniform_(-3, 3).item() / 20)
            waveform = waveform * gain

        # time shift (circular)
        if torch.rand(1).item() < 0.3:
            shift = int(torch.randint(0, int(0.1 * self.target_sr), (1,)).item())
            waveform = torch.roll(waveform, shift)

        return waveform

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        meta = self.samples[idx]

        waveform = self._load_audio(meta["audio_path"])
        waveform = self._pad_or_crop(waveform)

        if self.augment:
            waveform = self._augment(waveform)

        # normalise to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        item = {"waveform": waveform}

        if self.label_mode == "joint":
            item["label"] = torch.tensor(meta["joint_idx"], dtype=torch.long)
        else:  # two_head
            item["emotion_label"] = torch.tensor(meta["emotion_idx"], dtype=torch.long)
            item["level_label"]   = torch.tensor(meta["level_idx"],   dtype=torch.long)

        return item

    # ── class-weight helpers ──────────────────────────────────────────────────

    def get_class_weights(self) -> dict:
        """
        Returns inverse-frequency weights for use in CrossEntropyLoss or
        WeightedRandomSampler.

        Returns dict with keys depending on label_mode:
          joint    ==> {"joint": Tensor(NUM_JOINT_CLASSES)}
          two_head ==> {"emotion": Tensor(6), "level": Tensor(3)}
        """
        from collections import Counter

        if self.label_mode == "joint":
            counts = Counter(s["joint_idx"] for s in self.samples)
            weights = torch.zeros(NUM_JOINT_CLASSES)
            for cls, cnt in counts.items():
                weights[cls] = 1.0 / cnt
            weights = weights / weights.sum()
            return {"joint": weights}
        else:
            e_counts = Counter(s["emotion_idx"] for s in self.samples)
            l_counts = Counter(s["level_idx"]   for s in self.samples)

            e_w = torch.zeros(NUM_EMOTION_CLASSES)
            for cls, cnt in e_counts.items():
                e_w[cls] = 1.0 / cnt
            e_w = e_w / e_w.sum()

            l_w = torch.zeros(NUM_LEVEL_CLASSES)
            for cls, cnt in l_counts.items():
                l_w[cls] = 1.0 / cnt
            l_w = l_w / l_w.sum()

            return {"emotion": e_w, "level": l_w}

    def get_sampler_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler (joint label used)."""
        from collections import Counter
        counts = Counter(s["joint_idx"] for s in self.samples)
        weights = torch.tensor(
            [1.0 / counts[s["joint_idx"]] for s in self.samples],
            dtype=torch.float,
        )
        return weights


# ── collate (handles variable-length if max_duration is disabled) ─────────────

def collate_fn(batch):
    """Pads waveforms in a batch to the same length."""
    max_len = max(item["waveform"].shape[0] for item in batch)
    waveforms = torch.stack([
        torch.nn.functional.pad(item["waveform"], (0, max_len - item["waveform"].shape[0]))
        for item in batch
    ])
    out = {"waveform": waveforms}

    if "label" in batch[0]:
        out["label"] = torch.stack([item["label"] for item in batch])
    if "emotion_label" in batch[0]:
        out["emotion_label"] = torch.stack([item["emotion_label"] for item in batch])
        out["level_label"]   = torch.stack([item["level_label"]   for item in batch])
    return out

def collate_fn_synthesized(batch):
    """Pads waveforms in a batch to the same length."""
    max_len = max(item["waveform"].shape[0] for item in batch)

    waveforms = torch.stack([
        torch.nn.functional.pad(
            item["waveform"],
            (0, max_len - item["waveform"].shape[0])
        )
        for item in batch
    ])

    out = {"waveform": waveforms}

    # ── labels ──
    if "label" in batch[0]:
        out["label"] = torch.stack([item["label"] for item in batch])

    if "emotion_label" in batch[0]:
        out["emotion_label"] = torch.stack([item["emotion_label"] for item in batch])
        out["level_label"]   = torch.stack([item["level_label"]   for item in batch])

    # ── keep filenames ──
    if "filename" in batch[0]:
        out["filename"] = [item["filename"] for item in batch]

    return out


# ── quick test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "train_filelist.txt"
    ds = MEADDataset(path, augment=True, label_mode="two_head")
    item = ds[0]
    print("waveform shape :", item["waveform"].shape)
    print("emotion label  :", item["emotion_label"].item(),
          "==>", EMOTIONS[item["emotion_label"].item()])
    print("level label    :", item["level_label"].item(),
          "==>", LEVELS[item["level_label"].item()])

    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print("batch waveform :", batch["waveform"].shape)
