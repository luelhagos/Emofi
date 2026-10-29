# output/ckpt/

Checkpoints are loaded as `<ckpt_path>/<step>.pth.tar`, where `ckpt_path` comes
from the train config (`output/ckpt/MEAD` for the backbone, `output/ckpt/MEAD_intensity`
for the refinement stage). `jobs/train.sh` warm-starts from
`output/ckpt/pretrained/warmstart.pth.tar`; override the path with
`PRETRAINED=/path/to/ckpt.pth.tar bash jobs/train.sh 0`.

Pretrained checkpoints are hosted on the Hugging Face Hub (`luel/emofi-tts`);
see the top-level README for the download commands and placement.
Training writes its checkpoints into the `ckpt_path` directories above; use a
different `ckpt_path` in the train config to keep several runs side by side.
