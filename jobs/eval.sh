#!/bin/bash
# Evaluation: synthesize the test set, then score WER + emotion2vec.
#
# Usage:  bash jobs/eval.sh [GPU_ID]
# Prereq: a trained checkpoint at ./output/ckpt/MEAD/<STEP>.pth.tar (set STEP),
#         data under ./data/ (see README).
set -e
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=${1:-0}

DATA=data/MEAD                 # preprocessed data (test.txt lives here)
REAL=data/MEAD_processed       # real test wavs (for emotion2vec)
STEP=${STEP:-30000}            # checkpoint step (= total_step in train.yaml)
OUT=eval/test

mkdir -p eval

echo "=== [1/3] synthesize the test set -> $OUT ==="
python3 scripts/synthesize_test_set.py --restore_step $STEP \
    -p config/MEAD/preprocess.yaml -m config/MEAD/model.yaml -t config/MEAD/train.yaml \
    --manifest test.txt --combine_mode concat_linear --speaker_emb --out_dir "$OUT"

echo "=== [2/3] intelligibility: WER (Whisper) ==="
python3 scripts/eval_wer.py --wav_dir "$OUT" --manifest "$DATA/test.txt" \
    --out_txt eval/wer.txt

echo "=== [3/3] emotion: emotion2vec judge ==="
python3 scripts/eval_emotion2vec.py --real_dir "$REAL" --manifest "$DATA/test.txt" \
    --synth_dirs "$OUT" --out_txt eval/emotion2vec.txt

echo
echo "done -> eval/wer.txt , eval/emotion2vec.txt"
