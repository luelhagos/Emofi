# Emofi-TTS

Face-conditioned emotional text-to-speech with disentangled control: a **face
image** drives the emotion and its intensity, a **reference utterance** drives the
speaker identity, and **text** drives the content. The three inputs meet in a single
linearly-projected style vector on a FastSpeech2 / Meta-StyleSpeech backbone; an
intensity-conditioned refinement module trained by SER feature matching restores
graded intensity; HiFi-GAN vocodes the result.

For checkpoints and processed data, check this link: https://drive.google.com/drive/folders/1AMTuWvHln3T7sZAgLrt_7k8rKNE9Gm5n?usp=drive_link

## Repository layout

```
train.py          training entry point (backbone stage and refinement stage)
dataset.py        data loader: mels / pitch / energy / durations + cached face & speaker embeddings,
                  cross-speaker face sampling
evaluate.py       validation loop run periodically during training
prepare_align.py  stage MEAD wavs + transcripts for forced alignment
preprocess.py     extract mels / pitch / energy / durations from the aligned data (preprocessor/, audio/)

model/            StyleSpeech backbone (SALN), variance adaptor, losses,
                  postfilter.py (refinement module + mel discriminators), serfm.py (SER feature matching)
utils/            checkpoint & vocoder loading (model.py); tensors / masks (tools.py)
text/             text front end (phoneme symbols, text-to-sequence; the symbol table is fixed by the trained checkpoints)
hifigan/          HiFi-GAN vocoder code (generator weights are not included)
lexicon/          grapheme-to-phoneme lexicon (only needed to phonemize new free-form text)
face_encoder/     face encoder model definition + inference (see face_encoder/README.md)
ser_judge/        Wav2Vec2 SER judge -> emotion + intensity WA/UA/per-level recall (own env; see ser_judge/README.md)

config/MEAD/      preprocess.yaml (data & features); model.yaml + train.yaml (backbone);
                  model_intensity.yaml + train_intensity.yaml (refinement stage)
jobs/             one-command runners: train.sh, train_intensity.sh, eval.sh, eval_intensity.sh

scripts/          synthesis
                    synthesize_test_set.py     synthesize the test set (refinement / level flags)
                    synthesize_swap_test.py    cross-identity swap synthesis (+ --zero_face / --zero_ref controls)
                  evaluation
                    eval_wer.py                Whisper-large WER / CER (jiwer)
                    eval_emotion2vec.py        emotion2vec emotion WA/UA (independent judge)
                    eval_utmos.py              UTMOS naturalness predictor
                    eval_arousal.py            continuous arousal (dimensional judge) + per-level range
                    speaker_transfer_probe.py  speaker transfer % on swap outputs (ECAPA nearest-centroid)
                    eval_speaker_similarity.py speaker similarity to the real recording (Resemblyzer cosine) + real-pair anchors
                    check_ref_arousal_bleed.py reference-arousal leakage r on swap outputs
                    eval_intensity_probe.py    held-out intensity judge (WavLM probe fit on real audio only)
                  embeddings / disentanglement
                    cache_face_embeddings.py   face image -> 768-d embedding cache (uses face_encoder/)
                    cache_speaker_embeddings.py reference wav -> 192-d ECAPA embedding cache
                    leace_disentangle.py       LEACE speaker erasure on face embeddings + linear/MLP probes
                    probe_ref_emotion_leak.py  emotion probes on ECAPA embeddings + category erasure
                    leace_arousal_strip.py     continuous-arousal erasure on ECAPA embeddings
                  data preparation
                    match_canonical.py         content-match transcripts to MEAD's canonical sentence list
                    whisper_verify.py          confirm uncertain matches with Whisper-large on the raw audio
                    prepare_realign.py         write corrected transcripts for forced re-alignment
```


## Setup

**1. Environment**
```bash
pip install -r requirements.txt
```
The SER judge runs in its own environment (`ser_judge/requirements.txt`).

**2. Models and data.** Pretrained checkpoints and the preprocessed MEAD package are
not distributed with this repository yet; they will be released with the paper. The
code expects:

| Path | Contents |
|---|---|
| `data/MEAD/` | preprocessed features (mel, pitch, energy, duration), cached face + speaker embeddings, manifests, `stats.json` |
| `data/MEAD_processed/` | real test wavs (real-audio ceilings; reference clips) |
| `output/ckpt/pretrained/warmstart.pth.tar` | warm start for `jobs/train.sh` |
| `output/ckpt/MEAD/<step>.pth.tar` | backbone checkpoints (written by `jobs/train.sh`) |
| `output/ckpt/MEAD_intensity/<step>.pth.tar` | refinement checkpoints (written by `jobs/train_intensity.sh`) |
| `hifigan/generator_universal.pth.tar` | HiFi-GAN vocoder weights |
| `face_encoder/face_encoder.pt` | face encoder weights (only needed to embed new face images) |
| `ser_judge/best_model.pt` | SER judge weights (evaluation only) |

With the embedding caches inside `data/MEAD/`, neither encoder is needed to train or
evaluate on MEAD.


## Evaluate a trained system

Synthesize the test set with a trained system in the image-only condition
(emotion and intensity level both predicted from the face), then score it.
`OUT` is any directory for the synthesized wavs.

```bash
OUT=<output_dir>

# 1. synthesize (refinement applied at levels 2/3; level 1 bypasses)
python3 scripts/synthesize_test_set.py --restore_step 15000 \
    -p config/MEAD/preprocess.yaml -m config/MEAD/model_intensity.yaml -t config/MEAD/train_intensity.yaml \
    --manifest test.txt --combine_mode concat_linear --speaker_emb \
    --postfilter_refine --predict_level --postfilter_levels level2,level3 \
    --out_dir $OUT

# 2. score
python3 ser_judge/evaluate_synthesized.py --checkpoint ser_judge/best_model.pt \
    --wav_dir $OUT --output_txt $OUT/ser.txt        # emotion WA + intensity WA/UA + L1/L2/L3 (SER-judge env)
python3 scripts/eval_wer.py   --wav_dir $OUT --manifest data/MEAD/test.txt --out_txt $OUT/wer.txt
python3 scripts/eval_utmos.py --wav_dirs $OUT --out_tsv $OUT/utmos.tsv
python3 scripts/eval_emotion2vec.py --real_dir data/MEAD_processed --manifest data/MEAD/test.txt \
    --synth_dirs $OUT --out_txt $OUT/e2v.txt
```
Score the real test audio (`data/MEAD_processed`) with the same commands to obtain
the reference ceilings for each metric.

Speaker transfer (cross-identity swaps):
```bash
SWAP=<swap_output_dir>
python3 scripts/synthesize_swap_test.py --restore_step 15000 \
    -p config/MEAD/preprocess.yaml -m config/MEAD/model_intensity.yaml -t config/MEAD/train_intensity.yaml \
    --manifest test.txt --combine_mode concat_linear --speaker_emb \
    --postfilter_refine --predict_level --postfilter_levels level2,level3 --out_dir $SWAP
python3 scripts/speaker_transfer_probe.py --swap_dir $SWAP \
    --pairs ${SWAP}_pairs.txt --real_dir data/MEAD_processed
```
Add `--zero_face` or `--zero_ref` to the swap synthesis for the input-ablation
controls; score the swap outputs with the SER judge against both the face's and
the reference's emotion label for the factorization test.


## Train

**Stage 1: backbone** (cross-speaker training, 30k steps, warm-started from a pretrained multi-speaker checkpoint)
```bash
bash jobs/train.sh 0          # arg = GPU id; checkpoints -> output/ckpt/MEAD/
```
**Stage 2: refinement module** on the frozen backbone (20k steps; the 15k
checkpoint is used)
```bash
BACKBONE=output/ckpt/MEAD/30000.pth.tar bash jobs/train_intensity.sh 0   # -> output/ckpt/MEAD_intensity/
```
Quick evaluation of your own checkpoints: `STEP=30000 bash jobs/eval.sh 0`
(backbone: WER + emotion2vec) and `STEP=15000 PREDICT_LEVEL=1 bash jobs/eval_intensity.sh 0`
(refinement, image-only: SER + WER). `PREDICT_LEVEL=2` runs the fully continuous
mode (expected level from the face head, `--predict_level_expected --pf_ramp`).


## Further evaluations

- **Continuous intensity dial**: `synthesize_test_set.py --level_interp <t> --pf_ramp`
  (t in [0, 2] maps to intensity levels 1..3); score each setting with the SER judge.
- **Speaker similarity**: `eval_speaker_similarity.py --real_dir data/MEAD_processed --synth_dirs $OUT`
  scores each synthesized utterance against the real recording of the same utterance
  (Resemblyzer d-vector cosine), with same-speaker and different-speaker real pairs as
  anchors; `--swap_dir $SWAP --pairs_file ${SWAP}_pairs.txt` scores swap outputs
  against the reference speaker's centroid.
- **Held-out intensity judge**: `eval_intensity_probe.py` fits a WavLM probe on real
  training audio only and scores synthesized directories.
- **Arousal and leakage**: `eval_arousal.py` (continuous arousal, per-level range);
  `check_ref_arousal_bleed.py` (does the output's arousal track the reference's?).
- **Closed-form erasure (LEACE)**: `probe_ref_emotion_leak.py` (emotion probes on the
  ECAPA cache; `--erased_out` writes the category-erased cache),
  `leace_arousal_strip.py` (continuous-arousal erasure), `leace_disentangle.py`
  (speaker probes / erasure on the face cache). Point `synthesize_swap_test.py` at an
  erased cache via `--spk_emb_dir`.


## Preparing MEAD from scratch

To build `data/MEAD/` from MEAD:

1. Extract per-utterance wavs and a frontal face frame per utterance into
   `data/MEAD_corpus/` as `<speaker>_<emotion>_<level>_<index>.wav` with a
   transcript `<utt>_CorrectedTranscript.txt` beside each wav.
2. Stage the wavs and transcripts for forced alignment, then run the Montreal
   Forced Aligner and place its TextGrids under `data/MEAD/TextGrid/`:
   ```bash
   python3 prepare_align.py config/MEAD/preprocess.yaml          # -> data/MEAD_align/<speaker>/
   mfa align data/MEAD_align lexicon/librispeech-lexicon.txt english <mfa_out>
   python3 preprocessor/mead.py nest_textgrids --flat <mfa_out> --nested data/MEAD/TextGrid
   ```
3. Extract mels, pitch, energy, and durations, and write the manifests:
   ```bash
   python3 preprocess.py config/MEAD/preprocess.yaml
   ```
4. Repair the transcripts if the source transcripts are ASR-generated
   (`match_canonical.py` -> `whisper_verify.py` -> `prepare_realign.py`, then
   re-align and re-run step 3 with the corrected text).
5. Cache the embeddings: `scripts/cache_face_embeddings.py` (needs
   `face_encoder/face_encoder.pt`) and `scripts/cache_speaker_embeddings.py`
   (ECAPA-TDNN via SpeechBrain).
