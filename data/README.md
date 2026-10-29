# data/

Download the data package from the gated Hugging Face dataset repository
`luel/emofi-mead` (see the top-level README). The code expects two folders here:

- `data/MEAD/`            preprocessed training data: `mel/ pitch/ energy/ duration/
                          face_emb_cache/ spk_emb_cache/ speakers.json stats.json
                          train.txt val.txt test.txt` (built by `preprocess.py`, see the top-level README)
- `data/MEAD_processed/`  real test wavs (real-audio ceilings for the emotion2vec / SER judges; reference clips)
