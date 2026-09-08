"""
Cache face-encoder embeddings for every MEAD utterance.

Reads the face-encoder filelists (rows `<audio>|<face>|...`), runs the released
face encoder (face_encoder/) on each utterance's paired face image, and writes
one L2-normalized (768,) tensor per utterance basename. Existing files are
skipped. The TTS dataset loads these instead of running the encoder per batch.

Layout produced:
    <out_dir>/<basename>.pt

Usage:
    python3 scripts/cache_face_embeddings.py \
        --checkpoint   face_encoder/face_encoder.pt \
        --filelist_dir filelists \
        --image_root   data/MEAD_processed \
        --out_dir      data/MEAD/face_emb_cache
"""

import argparse
import os
import sys

import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from face_encoder.infer import FaceEmotionInference  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--face_enc_dir", default=None,
                   help="Unused; kept for backward compatibility (the encoder "
                        "is imported from face_encoder/).")
    p.add_argument("--checkpoint",
                   default=os.path.join(REPO_ROOT, "face_encoder", "face_encoder.pt"),
                   help="Face-encoder checkpoint with model_state + model_config.")
    p.add_argument("--filelist_dir", required=True,
                   help="Dir containing {train,val,test}.txt of the face encoder.")
    p.add_argument("--image_root",   required=True,
                   help="Root dir where face PNGs live (filelist face paths "
                        "are joined with this when not absolute).")
    p.add_argument("--out_dir",      required=True,
                   help="Where to write <basename>.pt files.")
    p.add_argument("--device",       default="cuda")
    return p.parse_args()


def build_audio_to_face(filelist_dir, image_root):
    """Read all three splits; return basename -> resolved face image path."""
    mapping = {}
    for split in ("train.txt", "val.txt", "test.txt"):
        path = os.path.join(filelist_dir, split)
        if not os.path.exists(path):
            print(f"[warn] missing {path}, skipping"); continue
        with open(path) as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) < 2:
                    continue
                audio_path, face_path = parts[0], parts[1]
                base = os.path.splitext(os.path.basename(audio_path))[0]
                if not os.path.isabs(face_path):
                    face_path = os.path.join(image_root, os.path.basename(face_path))
                mapping[base] = face_path
    return mapping


def main():
    args = parse_args()

    args.checkpoint   = os.path.expandvars(os.path.expanduser(args.checkpoint))
    args.filelist_dir = os.path.expandvars(os.path.expanduser(args.filelist_dir))
    args.image_root   = os.path.expandvars(os.path.expanduser(args.image_root))
    args.out_dir      = os.path.expandvars(os.path.expanduser(args.out_dir))

    os.makedirs(args.out_dir, exist_ok=True)

    audio_to_face = build_audio_to_face(args.filelist_dir, args.image_root)
    print(f"loaded {len(audio_to_face)} audio->face pairings")

    inf = FaceEmotionInference(args.checkpoint, device=args.device)

    written = already = missing_face = errored = 0
    for base, face_path in tqdm(audio_to_face.items()):
        out_path = os.path.join(args.out_dir, f"{base}.pt")
        if os.path.exists(out_path):
            already += 1
            continue
        if not os.path.exists(face_path):
            missing_face += 1
            continue
        try:
            emb = inf.get_embedding(face_path).cpu()   # (768,), L2-normalized
            torch.save(emb, out_path)
            written += 1
        except Exception as e:
            print(f"[err] {base}: {e}")
            errored += 1

    print(
        f"\nDone. wrote {written}, already had {already}, "
        f"missing face image {missing_face}, errored {errored}"
    )


if __name__ == "__main__":
    main()
