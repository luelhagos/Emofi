# output/ckpt/pretrained/

Place the warm-start checkpoint here, from the [trivox-tts](https://drive.google.com/drive/folders/1AMTuWvHln3T7sZAgLrt_7k8rKNE9Gm5n?usp=drive_link) Google Drive:

- `warmstart.pth.tar`   pretrained backbone used to warm-start training (`jobs/train.sh`)

Training writes its own checkpoints to `output/ckpt/MEAD/` (created automatically).
To override the warm-start path without using this folder:
`PRETRAINED=/path/to/ckpt.pth.tar bash jobs/train.sh 0`.
