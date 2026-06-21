import os
import json

import torch
import numpy as np

import hifigan
from model import StyleSpeech, ScheduledOptimMain, ScheduledOptimDisc


def load_pretrained(model, ckpt_path):
    """Load a foreign checkpoint into `model`, skipping keys that don't exist in
    the target or have a shape mismatch. Used to warm-start from a base checkpoint
    when the target model has new layers (face_to_style, emotion_aux_head) and a
    different style_prototype size.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model", ckpt)   # supports {"model": sd} or a raw state_dict

    tgt = model.state_dict()
    loaded, skipped_missing, skipped_mismatch = [], [], []
    new_sd = {}
    for k, v in src.items():
        if k not in tgt:
            skipped_missing.append(k)
            continue
        if tgt[k].shape != v.shape:
            skipped_mismatch.append((k, tuple(v.shape), tuple(tgt[k].shape)))
            continue
        new_sd[k] = v
        loaded.append(k)

    missing_in_src = [k for k in tgt if k not in src]

    result = model.load_state_dict(new_sd, strict=False)
    print(f"[load_pretrained] {ckpt_path}")
    print(f"  loaded:               {len(loaded)} keys")
    print(f"  skipped (not in tgt): {len(skipped_missing)} keys")
    for k in skipped_missing[:5]:
        print(f"    - {k}")
    if len(skipped_missing) > 5:
        print(f"    ... +{len(skipped_missing)-5} more")
    print(f"  skipped (shape mismatch): {len(skipped_mismatch)} keys")
    for k, ss, ts in skipped_mismatch:
        print(f"    - {k}: ckpt {ss} -> target {ts}")
    print(f"  not in checkpoint (kept random init): {len(missing_in_src)} keys")
    for k in missing_in_src[:10]:
        print(f"    - {k}")
    if len(missing_in_src) > 10:
        print(f"    ... +{len(missing_in_src)-10} more")
    return result


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = StyleSpeech(preprocess_config, model_config).to(device)

    ckpt = None
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, weights_only=False)
        # strict=False so the model can load a checkpoint that lacks some of its
        # parameters; any missing params keep their initialization. Report what differed.
        result = model.load_state_dict(ckpt["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"[restore {args.restore_step}] strict=False -- "
                  f"missing={list(result.missing_keys)} "
                  f"unexpected={list(result.unexpected_keys)}")
    elif getattr(args, "pretrained_ckpt", None):
        # Warm-start from a base checkpoint.
        load_pretrained(model, args.pretrained_ckpt)

    if train:
        scheduled_optim_main = ScheduledOptimMain(
            model, train_config, model_config, args.restore_step,
            extra_filters=None,
        )
        scheduled_optim_disc = ScheduledOptimDisc(
            model, train_config
        )
        if args.restore_step and ckpt is not None:
            scheduled_optim_main.load_state_dict(ckpt["optimizer_main"])
            scheduled_optim_disc.load_state_dict(ckpt["optimizer_disc"])
        model.train()
        return model, scheduled_optim_main, scheduled_optim_disc

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

