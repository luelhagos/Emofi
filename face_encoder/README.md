# Face encoder (inference only)

The face encoder maps a single face image to a 768-d embedding that conditions
the TTS model. It is an ImageNet-pretrained ResNet-50 trained to align with
emotion2vec audio emotion representations, jointly with auxiliary emotion and
intensity classification heads.

This folder contains the model definition and inference code only; training
code is not included.

## Checkpoint

Download `face_encoder.pt` from the Hugging Face model repository
(`hf download luel/emofi-tts face_encoder.pt --local-dir face_encoder`);
the code expects it at:

```
face_encoder/face_encoder.pt
```

Format: a dict with `model_config` (`embedding_dim`, `num_emotions`,
`num_intensities`), `model_state`, `epoch`, `val_loss`.

## Usage

Embed one face image:

```bash
python3 -c "from face_encoder.infer import FaceEmotionInference as F; \
  print(F('face_encoder/face_encoder.pt', device='cuda').get_embedding('face.png').shape)"
```

Build the `face_emb_cache` used by training / synthesis (one L2-normalized
`<basename>.pt` per utterance, keyed by the face-encoder filelists):

```bash
python3 scripts/cache_face_embeddings.py \
    --checkpoint   face_encoder/face_encoder.pt \
    --filelist_dir filelists \
    --image_root   data/MEAD_processed \
    --out_dir      data/MEAD/face_emb_cache
```

The preprocessed `data/MEAD/` package contains this cache, so the encoder is
only needed to embed new faces.

Files:

- `model.py` - `FaceEmotionEncoder` class (needed to load the checkpoint)
- `infer.py` - `FaceEmotionInference` wrapper (single image / batch / with labels)
- `extract_face_cache.py` - alternative cache builder that writes raw embeddings from filelists
