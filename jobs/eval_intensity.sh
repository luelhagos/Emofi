#!/bin/bash
# Evaluate the intensity postfilter: synthesize the test set through the gated
# postfilter (level2/level3 refined, level1 bypasses), score emotion + intensity
# with the SER judge, and WER.
#
# Usage:  bash jobs/eval_intensity.sh [GPU_ID]
#   PREDICT_LEVEL=1  -> image-only condition: the intensity level is predicted
#                       from the face instead of read from the filename (the
#                       true three-input setting; also prints the face->level
#                       prediction accuracy).
#   PREDICT_LEVEL=2  -> continuous image-only condition: the expected level
#                       from the face head's probabilities interpolates the
#                       postfilter level embedding, with the residual ramped in
#                       below t=0.5 (--predict_level_expected --pf_ramp).
# Prereq: a postfilter checkpoint at ./output/ckpt/MEAD_intensity/<STEP>.pth.tar,
#         the SER judge checkpoint (see ser_judge/README.md), data under ./data/.
set -e
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=${1:-0}

DATA=data/MEAD
REAL=data/MEAD_processed
STEP=${STEP:-15000}
OUT=eval/intensity_s${STEP}
FLAGS="--postfilter_refine --postfilter_levels level2,level3"
if [ "${PREDICT_LEVEL:-0}" = "1" ]; then
    FLAGS="$FLAGS --predict_level"
    OUT=${OUT}_imageonly
elif [ "${PREDICT_LEVEL:-0}" = "2" ]; then
    FLAGS="$FLAGS --predict_level_expected --pf_ramp"
    OUT=${OUT}_imageonly_expected
fi

mkdir -p eval

echo "=== [1/3] synthesize the test set (gated postfilter) -> $OUT ==="
python3 scripts/synthesize_test_set.py --restore_step $STEP \
    -p config/MEAD/preprocess.yaml -m config/MEAD/model_intensity.yaml \
    -t config/MEAD/train_intensity.yaml \
    --manifest test.txt --combine_mode concat_linear --speaker_emb \
    $FLAGS --out_dir "$OUT"

echo "=== [2/3] emotion + per-level intensity (SER judge) ==="
SER_CKPT=${SER_CKPT:-ser_judge/best_model.pt}
python3 ser_judge/evaluate_synthesized.py --checkpoint "$SER_CKPT" \
    --wav_dir "$OUT" --output_txt eval/ser_intensity_s${STEP}.txt

echo "=== [3/3] intelligibility: WER (Whisper) ==="
python3 scripts/eval_wer.py --wav_dir "$OUT" --manifest "$DATA/test.txt" \
    --out_txt eval/wer_intensity_s${STEP}.txt

echo
echo "Also score the real test audio once (the per-level ceiling), and the"
echo "no-postfilter baseline (rerun without --postfilter_refine) for the strict"
echo "same-checkpoint comparison."
