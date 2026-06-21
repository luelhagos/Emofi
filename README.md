# TriVox-TTS

Face-conditioned emotional text-to-speech with disentangled control: a **face
image** drives the emotion, a **reference utterance** drives the speaker identity,
and **text** drives the content. The three meet in a single style bottleneck on a
FastSpeech2 / Meta-StyleSpeech backbone and are vocoded with HiFi-GAN.


## Repository layout

```
train.py          main training entry point
dataset.py        training data loader (mels / pitch / energy / durations + cached face & speaker embeddings)
evaluate.py       validation loop run periodically during training

model/            TTS architecture: StyleSpeech backbone, variance adaptor, SALN, loss modules
utils/            helpers: checkpoint & vocoder loading (model.py); tensors / masks / plotting (tools.py)
text/             text front end: phoneme symbol set and text-to-sequence conversion
hifigan/          HiFi-GAN neural vocoder, mel -> waveform (code only; generator weights from the Drive)
lexicon/          grapheme-to-phoneme lexicon (only needed to phonemize new, free-form text)

config/MEAD/      preprocess.yaml (data & features), model.yaml (architecture), train.yaml (optimizer & steps)
jobs/             one-command runners: train.sh (train), eval.sh (synthesize + WER + emotion2vec)
scripts/          eval & synthesis helpers, run individually:
                    synthesize_test_set.py   synthesize the test set to wavs
                    synthesize_swap_test.py  cross-speaker swap synthesis (factorization probe)
                    eval_wer.py              Whisper word error rate (intelligibility)
                    eval_emotion2vec.py      emotion accuracy + similarity (emotion2vec judge)
ser_judge/        Wav2Vec2 SER judge -> Emotion + Intensity WA/UA (separate env; see ser_judge/README.md)
```

## Setup

**1. Environment**
```bash
pip install -r requirements.txt
```

**2. Download from the [trivox-tts](https://drive.google.com/drive/folders/1AMTuWvHln3T7sZAgLrt_7k8rKNE9Gm5n?usp=drive_link) Google Drive** and place the files as below
(paths are relative to the repo root):

| Drive artifact | Place at | Used for |
|---|---|---|
| `MEAD/` (preprocessed) | `data/MEAD/` | training data (mels, durations, pitch, energy, **cached face + speaker embeddings**, manifests) |
| `MEAD_processed/` | `data/MEAD_processed/` | real test wavs (emotion2vec / SER real-audio ceiling) |
| `warmstart.pth.tar` | `output/ckpt/pretrained/warmstart.pth.tar` | pretrained checkpoint to warm-start training |
| `generator_universal.pth.tar` | `hifigan/` | HiFi-GAN vocoder weights |
| `best_model.pt` | `ser_judge/` | SER judge weights (emotion + intensity eval) |

The face embeddings are already cached inside `data/MEAD/`, so the face encoder is
**not** required to train -- the encoder model is on the Drive separately if you want
to encode new faces.

## Train

```bash
bash jobs/train.sh 0          # arg = GPU id
```
Checkpoints are written to `output/ckpt/MEAD/`. To override the warm-start path:
`PRETRAINED=/path/to/ckpt.pth.tar bash jobs/train.sh 0`.

## Evaluate

```bash
STEP=30000 bash jobs/eval.sh 0
# -> eval/wer.txt          (Whisper WER -- intelligibility)
# -> eval/emotion2vec.txt  (emotion2vec -- emotion WA/UA on synth vs. the real-audio ceiling)
```
`STEP` is the checkpoint step to evaluate (defaults to the final training step). To
skip training entirely, download the optional trained checkpoint and run only this.

**Emotion + intensity (SER judge).** Run separately, in its own environment (see
`ser_judge/README.md`):
```bash
python ser_judge/evaluate_synthesized.py \
    --checkpoint ser_judge/best_model.pt --wav_dir eval/test --output_txt eval/ser.txt
# -> Emotion WA/UA + Intensity WA/UA
```


