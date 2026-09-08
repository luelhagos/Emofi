#!/bin/bash
# Train the TTS backbone, warm-started from the pretrained checkpoint.
#
# Usage:  bash jobs/train.sh [GPU_ID]
# Prereq: data under ./data/ and the warm-start checkpoint placed at $PRETRAINED
#         (see README). Checkpoints are written to ./output/ckpt/MEAD/.
#
# Cross-speaker training: with probability 0.5 the face embedding is taken from
# a different actor's utterance with the same emotion and intensity level, while
# the speaker embedding stays the target's own.

cd "$(dirname "$0")/.." || exit 1

PRETRAINED=${PRETRAINED:-output/ckpt/pretrained/warmstart.pth.tar}

CUDA_VISIBLE_DEVICES=${1:-0} python3 train.py \
    -p config/MEAD/preprocess.yaml \
    -m config/MEAD/model.yaml \
    -t config/MEAD/train.yaml \
    --pretrained_ckpt "$PRETRAINED" \
    --restore_step 0 \
    --combine_mode concat_linear \
    --speaker_emb \
    --lambda_emotion 0.1 \
    --lambda_intensity 0.1 \
    --lambda_adv 0.05 \
    --adv_warmup_steps 1000 \
    --mismatch_face \
    --mismatch_face_prob 0.5
