"""
Build a face-embedding cache (<out_dir>/<stem>.pt) from a face-encoder
checkpoint, using face-encoder filelists (audio_path|face_path|...).

Reads every filelist row, runs the encoder on the paired face image, and
writes one raw (un-normalized) embedding tensor per utterance stem. Existing
files are skipped.

Usage:
    python3 face_encoder/extract_face_cache.py \
        --ckpt face_encoder/face_encoder.pt \
        --filelists filelists/train.txt filelists/val.txt filelists/test.txt \
        --out_dir data/MEAD/face_emb_cache
"""

import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

try:
    from face_encoder.model import FaceEmotionEncoder
except ImportError:  # run from inside face_encoder/
    from model import FaceEmotionEncoder


def get_face_transform(image_size=224):
    """Evaluation-time image transform (resize + ImageNet normalization)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--filelists", nargs="+", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    dim = ck.get("config", {}).get("embedding_dim", 768)
    model = FaceEmotionEncoder(embedding_dim=dim).to(args.device).eval()
    model.load_state_dict(ck["model_state"])
    print(f"[extract] checkpoint epoch {ck.get('epoch', '?')}, dim {dim}")

    out = os.path.expandvars(os.path.expanduser(args.out_dir))
    os.makedirs(out, exist_ok=True)
    tf = get_face_transform(224)

    todo = []
    for fl in args.filelists:
        with open(os.path.expandvars(os.path.expanduser(fl))) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 6:
                    continue
                stem = os.path.splitext(os.path.basename(parts[0]))[0]
                dst = os.path.join(out, f"{stem}.pt")
                if not os.path.exists(dst):
                    todo.append((parts[1], dst))
    print(f"[extract] {len(todo)} embeddings to write -> {out}")

    with torch.no_grad():
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            imgs = torch.stack([tf(Image.open(p).convert("RGB"))
                                for p, _ in chunk]).to(args.device)
            embs = model.get_emotion_embedding(imgs).cpu()
            for (_, dst), e in zip(chunk, embs):
                torch.save(e.clone(), dst)
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{len(todo)}")
    print(f"[extract] done -> {out}")


if __name__ == "__main__":
    main()
