#!/bin/bash
# Train the intensity postfilter on a frozen, pretrained backbone.
#
# Usage:  bash jobs/train_intensity.sh [GPU_ID]
# Prereq: a trained backbone checkpoint (jobs/train.sh) at $BACKBONE, data under
#         ./data/. Checkpoints are written to ./output/ckpt/MEAD_intensity/.
#
# The backbone is frozen; only the postfilter generator and discriminator train.
# The SER feature-matching loss vocodes each refined mel and matches its
# frozen-SSL features to the vocoded ground-truth mel, for all three
# intensity levels (--serfm_all_levels; without it only level3 utterances are
# matched). At inference apply the postfilter to level2/level3 only
# (see jobs/eval_intensity.sh); low-intensity utterances bypass it.

cd "$(dirname "$0")/.." || exit 1

BACKBONE=${BACKBONE:-output/ckpt/MEAD/30000.pth.tar}

CUDA_VISIBLE_DEVICES=${1:-0} python3 train.py \
    -p config/MEAD/preprocess.yaml \
    -m config/MEAD/model_intensity.yaml \
    -t config/MEAD/train_intensity.yaml \
    --pretrained_ckpt "$BACKBONE" \
    --restore_step 0 \
    --combine_mode concat_linear \
    --speaker_emb \
    --lambda_emotion 0 \
    --lambda_intensity 0 \
    --lambda_adv 0 \
    --lambda_postfilter 1.0 \
    --pf_lambda_fm 2.0 \
    --pf_lambda_gv 1.0 \
    --pf_lambda_con 0.5 \
    --lambda_serfm 10.0 \
    --serfm_model facebook/wav2vec2-base \
    --serfm_layer 4 \
    --serfm_all_levels \
    --freeze_backbone_for_postfilter
