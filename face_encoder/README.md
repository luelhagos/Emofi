# Face encoder (inference only)

The face-to-emotion encoder maps a single face image to an emotion embedding
that conditions the TTS model. Architecture: ResNet50 backbone + MLP projector
to a 768-d embedding, with auxiliary emotion and intensity classification heads,
distilled from emotion2vec (`iic/emotion2vec_plus_large`) audio embeddings.

The encoder was developed by a project collaborator; this folder contains the
model definition and inference code only. Training code is not included.

## Checkpoint

The encoder weights are not distributed with this repository yet. The code
expects them at:

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
