# EmoFI-TTS

Face-conditioned emotional text-to-speech with disentangled control. A single
**face image** sets the emotion and its intensity, a **reference utterance** sets
the speaker identity, and **text** sets the content. The face and speaker
embeddings meet only in one linearly projected style vector that conditions a
FastSpeech2 / Meta-StyleSpeech backbone; an intensity-conditioned refinement
module, trained by feature matching against a frozen speech encoder, restores
the graded intensity that regression decoders smooth away; HiFi-GAN produces the
waveform.

This repository contains the full system described in the paper: both training
stages, synthesis in the discrete and continuous-intensity modes, and every
evaluation behind the reported results.

- **Paper:** [EmoFI-TTS: Face-Conditioned Expressive Voice Synthesis with Intensity Modulation](https://arxiv.org/abs/2610.30477) (arXiv:2610.30477, 2026)
- **Weights:** [luel/emofi-tts](https://huggingface.co/luel/emofi-tts) on the Hugging Face Hub
- **Data:** [luel/emofi-mead](https://huggingface.co/datasets/luel/emofi-mead), preprocessed MEAD (gated, research use)

**Contents:**
[Installation](#installation) |
[Models and data](#models-and-data) |
[Synthesize and evaluate](#synthesize-and-evaluate) |
[Train](#train) |
[Further evaluations](#further-evaluations) |
[Preparing MEAD from scratch](#preparing-mead-from-scratch) |
[Repository layout](#repository-layout) |
[Pretrained components](#pretrained-components) |
[Citation](#citation)


## Installation

```bash
git clone https://github.com/luelhagos/Emofi.git
cd Emofi
pip install -r requirements.txt
```

The speech-emotion judge used for evaluation runs in its own environment
(`ser_judge/requirements.txt`, see `ser_judge/README.md`).


## Models and data

Weights are in the model repository `luel/emofi-tts`. The preprocessed MEAD
package is in the gated dataset repository `luel/emofi-mead`: accept the MEAD
research-use terms once on the Hub, then log in with `hf auth login`. From the
repository root, download everything into place with:

```bash
pip install -U huggingface_hub
hf download luel/emofi-tts  warmstart.pth.tar           --local-dir output/ckpt/pretrained
hf download luel/emofi-tts  backbone_30000.pth.tar      --local-dir output/ckpt/MEAD
hf download luel/emofi-tts  refinement_15000.pth.tar    --local-dir output/ckpt/MEAD_intensity
hf download luel/emofi-tts  generator_universal.pth.tar --local-dir hifigan
hf download luel/emofi-tts  face_encoder.pt             --local-dir face_encoder
hf download luel/emofi-tts  best_model.pt               --local-dir ser_judge
hf download luel/emofi-mead --repo-type dataset         --local-dir data
(cd data && unzip -q MEAD_features.zip && unzip -q MEAD_test_wavs.zip && mv MEAD_v23 MEAD && mv gt_test_wavs_v23 MEAD_processed)
mv output/ckpt/MEAD/backbone_30000.pth.tar output/ckpt/MEAD/30000.pth.tar
mv output/ckpt/MEAD_intensity/refinement_15000.pth.tar output/ckpt/MEAD_intensity/15000.pth.tar
```

Checkpoints are loaded as `<ckpt_path>/<step>.pth.tar`, so the two system
checkpoints are renamed to their training step. Extracting the features package
writes about 200k small files and can take 10 to 20 minutes on a network
filesystem. The resulting layout:

| Path | Contents |
|---|---|
| `output/ckpt/MEAD_intensity/15000.pth.tar` | released full system (backbone + refinement module) |
| `output/ckpt/MEAD/30000.pth.tar` | released backbone (after the cross-speaker stage); start point for `jobs/train_intensity.sh` |
| `output/ckpt/pretrained/warmstart.pth.tar` | backbone before the cross-speaker stage; warm start for `jobs/train.sh` |
| `hifigan/generator_universal.pth.tar` | HiFi-GAN vocoder weights |
| `face_encoder/face_encoder.pt` | face encoder weights (only needed to embed new face images) |
| `ser_judge/best_model.pt` | speech-emotion judge weights (evaluation only) |
| `data/MEAD/` | preprocessed features (mel, pitch, energy, duration), cached face + speaker embeddings, manifests, `stats.json` |
| `data/MEAD_processed/` | the real test utterances (real-audio ceilings; swap-test reference clips) |
| `data/MEAD_corpus.zip` | all per-utterance wavs with their original transcripts; only needed to rebuild the features from scratch |

Face and speaker embeddings are cached inside `data/MEAD/`, so neither encoder
is needed to train or evaluate on MEAD.


## Synthesize and evaluate

Synthesize the test set with the released system in the image-only condition
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
    --wav_dir $OUT --output_txt $OUT/ser.txt        # emotion WA + intensity WA/UA + L1/L2/L3 (judge env)
python3 scripts/eval_wer.py   --wav_dir $OUT --manifest data/MEAD/test.txt --out_txt $OUT/wer.txt
python3 scripts/eval_utmos.py --wav_dirs $OUT --out_tsv $OUT/utmos.tsv
python3 scripts/eval_emotion2vec.py --real_dir data/MEAD_processed --manifest data/MEAD/test.txt \
    --synth_dirs $OUT --out_txt $OUT/e2v.txt
```

Score the real test audio (`data/MEAD_processed`) with the same commands to obtain
the reference ceilings for each metric.

**Speaker transfer** (cross-identity swaps: the face from one speaker, the
reference utterance from another):

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
controls. For the factorization test, score the swap outputs with the
speech-emotion judge against both the face's and the reference's emotion label.


## Train

**Stage 1: backbone.** Cross-speaker training for 30k steps, warm-started from
`warmstart.pth.tar`:

```bash
bash jobs/train.sh 0          # arg = GPU id; checkpoints -> output/ckpt/MEAD/
```

**Stage 2: refinement module.** Trained on the frozen backbone for 20k steps;
the released system uses the 15k checkpoint:

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

`data/MEAD/` from the dataset repository is ready to use. To rebuild it, start
from `MEAD_corpus.zip` (per-utterance wavs and their original transcripts):

1. Unzip the corpus to `data/MEAD_corpus/` (`<speaker>_<emotion>_<level>_<index>.wav`
   with a `<utt>_CorrectedTranscript.txt` beside each wav) and add one frontal face
   frame per utterance.
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
4. Repair the transcripts: the shipped transcripts are ASR-generated and partly
   incomplete (`match_canonical.py` -> `whisper_verify.py` -> `prepare_realign.py`,
   then re-align and re-run step 3 with the corrected text).
5. Cache the embeddings: `scripts/cache_face_embeddings.py` (needs
   `face_encoder/face_encoder.pt`) and `scripts/cache_speaker_embeddings.py`
   (ECAPA-TDNN via SpeechBrain).


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
hifigan/          HiFi-GAN vocoder code
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


## Pretrained components

The system builds on publicly available models, all used frozen:
[ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (speaker
embeddings), [emotion2vec](https://github.com/ddlBoJack/emotion2vec) (face-encoder
distillation target and independent emotion judge),
[wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-base) (feature-matching
loss), [HiFi-GAN](https://github.com/jik876/hifi-gan) (vocoder),
[Whisper](https://github.com/openai/whisper) (word error rate),
[UTMOS](https://github.com/tarepan/SpeechMOS) (naturalness), and
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer) (speaker similarity).


## Acknowledgements

The backbone code derives from the public
[Meta-StyleSpeech](https://github.com/KevinMIN95/StyleSpeech) and
[FastSpeech2](https://github.com/ming024/FastSpeech2) implementations, and the
text front end from [Tacotron](https://github.com/keithito/tacotron). The face
encoder (`face_encoder/`) and the speech-emotion judge (`ser_judge/`) were
developed by Yejin Jeon, and are included with permission.
[MEAD](https://github.com/uniBruce/Mead) is used under its
research license; the preprocessed package is distributed under the same terms.


## License

The code in this repository is released under the MIT License (see `LICENSE`).
Model weights are released for research use; the MEAD-derived data package
follows the MEAD license.


## Citation

```bibtex
@article{jeon2026emofi,
  title   = {EmoFI-TTS: Face-Conditioned Expressive Voice Synthesis with Intensity Modulation},
  author  = {Jeon, Yejin and Beyene, Luel H. and Maltais, Marie and Kim, Hyounghun and Adelani, David I.},
  journal = {arXiv preprint arXiv:2610.30477},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.30477}
}
```
