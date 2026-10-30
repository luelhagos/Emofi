# SER judge (emotion + intensity)

A Wav2Vec2 speech-emotion classifier, fine-tuned on MEAD, used to score
synthesized speech for **emotion** (6-way) and **intensity** (3-way) accuracy.
This is an evaluation tool only.

## Setup

The judge depends on `transformers` (Wav2Vec2).

```bash
pip install -r ser_judge/requirements.txt
```

Download `best_model.pt` from the Hugging Face model repository
(`hf download luel/emofi-tts best_model.pt --local-dir ser_judge`);
the code expects it at `ser_judge/best_model.pt`.

## Run

```bash
python ser_judge/evaluate_synthesized.py \
    --checkpoint ser_judge/best_model.pt \
    --wav_dir    eval/test \
    --output_txt eval/ser.txt
```

Reports **Emotion WA/UA** and **Intensity WA/UA** over the wavs in `--wav_dir`.
Filenames must be `<speaker>_<emotion>_<level>_<index>.wav` (the format produced
by `scripts/synthesize_test_set.py`). Run it on the **real** test wavs too for the
judge's own ceiling.

Files: `model.py` (the classifier), `dataset.py` (feature extraction + labels),
`evaluate_synthesized.py` (the scorer).
