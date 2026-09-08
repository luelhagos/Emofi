# output/ckpt/

Checkpoints are loaded as `<ckpt_path>/<step>.pth.tar`, where `ckpt_path` comes
from the train config (`output/ckpt/MEAD` for the backbone, `output/ckpt/MEAD_intensity`
for the refinement stage). Place the released files from the
[trivox-tts](https://drive.google.com/drive/folders/1AMTuWvHln3T7sZAgLrt_7k8rKNE9Gm5n?usp=drive_link) Google Drive as:

| Drive file | place at | role |
|---|---|---|
| `warmstart.pth.tar` | `output/ckpt/pretrained/warmstart.pth.tar` | warm start for `jobs/train.sh` (backbone before the cross-speaker stage) |
| `backbone_30000.pth.tar` | `output/ckpt/MEAD/30000.pth.tar` | released backbone; start point for `jobs/train_intensity.sh` |
| `refinement_15000.pth.tar` | `output/ckpt/MEAD_intensity/15000.pth.tar` | released full system; `--restore_step 15000` with the `*_intensity.yaml` configs |

Training writes its own checkpoints into the same `ckpt_path` directories, so a
re-trained backbone at step 30000 overwrites the released one; use a different
`ckpt_path` in the train config to keep both. To override the warm-start path:
`PRETRAINED=/path/to/ckpt.pth.tar bash jobs/train.sh 0`.
