import argparse
import os

import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, get_mask_from_lengths
from model import MetaStyleSpeechLossMain, MetaStyleSpeechLossDisc
from dataset import Dataset

from evaluate import evaluate

# Optional wandb (no-op if not installed or not logged in).
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Emotion order -- matches the SER classifier used for evaluation.
EMOTION_MAP = {
    "angry":     0,
    "fear":      1,
    "happy":     2,
    "surprised": 3,
    "sad":       4,
    "neutral":   5,
}

# Intensity order -- matches the SER classifier levels.
INTENSITY_MAP = {
    "level1": 0,
    "level2": 1,
    "level3": 2,
}


def emotion_idx_from_ids(ids):
    """Parse '<spk>_<emo>_<lvl>_<idx>' -> emotion index tensor."""
    return torch.tensor(
        [EMOTION_MAP[name.split("_")[1]] for name in ids],
        dtype=torch.long,
    )


def intensity_idx_from_ids(ids):
    """Parse '<spk>_<emo>_<lvl>_<idx>' -> intensity index tensor."""
    return torch.tensor(
        [INTENSITY_MAP[name.split("_")[2]] for name in ids],
        dtype=torch.long,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def backward(model, optimizer, total_loss, step, grad_acc_step, grad_clip_thresh):
    total_loss = total_loss / grad_acc_step
    total_loss.backward()
    if step % grad_acc_step == 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
        optimizer.step_and_update_lr()
        optimizer.zero_grad()


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True,
        speaker_emb=args.speaker_emb,
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # >1 enables sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Model
    model, optimizer_main, optimizer_disc = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss_1 = MetaStyleSpeechLossMain(preprocess_config, model_config, train_config).to(device)
    Loss_2 = MetaStyleSpeechLossDisc(preprocess_config, model_config).to(device)
    print("Number of StyleSpeech Parameters:", num_param)

    # Vocoder
    vocoder = get_vocoder(model_config, device)

    # Logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # wandb (optional; enabled when WANDB_PROJECT is set)
    wandb_run = None
    if WANDB_AVAILABLE and os.environ.get("WANDB_PROJECT"):
        wandb_run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=os.environ.get("WANDB_RUN_NAME", "trivox"),
            config={
                "preprocess_config": preprocess_config,
                "model_config":      model_config,
                "train_config":      train_config,
                "restore_step":      args.restore_step,
            },
            resume="allow",
        )
        wandb.watch(model, log="gradients", log_freq=1000)

    # Training
    step = args.restore_step + 1
    epoch = 1
    meta_learning_warmup = train_config["step"]["meta_learning_warmup"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Face-conditioned generator path.
                if step <= meta_learning_warmup:
                    # batch layout:
                    #   0  ids               1  raw_texts
                    #   2  speakers          3  texts        4  text_lens     5  max_text_lens
                    #   6  mels              7  mel_lens     8  max_mel_lens
                    #   9  pitches          10  energies    11  durations
                    #  12  raw_quary_texts  13  quary_texts 14  quary_text_lens 15  max_quary_text_lens
                    #  16  quary_durations  17  face_embs   18  speaker embedding
                    face_in = batch[17]
                    output = (None, None, *model(
                        *(batch[2:12]),
                        face_embs=face_in,
                        ref_mels=None,
                        spk_embs=(batch[18] if args.speaker_emb else None),
                        combine_mode=args.combine_mode,
                    ))
                else:
                    # Step 1: Update Enc_s and G
                    output = model.module.meta_learner_1(*(batch[2:]))

                # Loss_1 unpacks inputs[6:] into 11 fields; drop face_embs (batch[17]).
                losses_1 = Loss_1(batch[:17], output)
                total_loss = losses_1[0]

                emotion_labels = None
                if args.lambda_emotion > 0:
                    emotion_labels = emotion_idx_from_ids(batch[0]).to(device)

                # Emotion-CE auxiliary loss on the style vector.
                emotion_ce_loss = None
                if args.lambda_emotion > 0:
                    aux_logits = model.module.compute_emotion_aux(batch[17])
                    emotion_ce_loss = F.cross_entropy(aux_logits, emotion_labels)
                    total_loss = total_loss + args.lambda_emotion * emotion_ce_loss

                # Intensity-CE auxiliary loss on the style vector.
                intensity_ce_loss = None
                if args.lambda_intensity > 0:
                    intensity_labels = intensity_idx_from_ids(batch[0]).to(device)
                    int_logits = model.module.compute_intensity_aux(batch[17])
                    intensity_ce_loss = F.cross_entropy(int_logits, intensity_labels)
                    total_loss = total_loss + args.lambda_intensity * intensity_ce_loss

                # Adversarial D_t loss on generated mels (G's fooling term). Linearly
                # ramp lambda_adv to avoid destabilizing an already-converged G.
                adv_g_loss = None
                effective_lambda_adv = 0.0
                if args.lambda_adv > 0:
                    ramp = min(1.0, max(0, step - args.restore_step) / max(1, args.adv_warmup_steps))
                    effective_lambda_adv = args.lambda_adv * ramp

                if effective_lambda_adv > 0:
                    gen_mel       = output[2]
                    gen_durations = output[6]
                    gen_mel_lens  = output[10]
                    max_gen_mel_len = int(max(gen_mel_lens).item())
                    gen_mel_masks   = get_mask_from_lengths(gen_mel_lens, max_gen_mel_len)
                    texts_emb       = model.module.phoneme_encoder.src_word_emb(batch[3])

                    D_t_fake_for_g = model.module.D_t(
                        model.module.variance_adaptor.upsample,
                        texts_emb, gen_mel,
                        max_gen_mel_len, gen_mel_masks, gen_durations,
                    )
                    adv_g_loss = F.mse_loss(
                        D_t_fake_for_g, torch.ones_like(D_t_fake_for_g)
                    )
                    total_loss = total_loss + effective_lambda_adv * adv_g_loss

                # Backward
                backward(model, optimizer_main, total_loss, step, grad_acc_step, grad_clip_thresh)

                # D_t update: zero its grads (leftover from adv_g_loss above), then
                # compute D_t loss on (real | fake) with detached inputs so grads do
                # not flow into G.
                d_t_loss = None
                if effective_lambda_adv > 0:
                    optimizer_disc.zero_grad()

                    real_mel        = batch[6]
                    real_durations  = batch[11]
                    real_mel_lens   = batch[7]
                    max_real_mel_len = int(batch[8])
                    real_mel_masks   = get_mask_from_lengths(real_mel_lens, max_real_mel_len)

                    texts_emb_d = model.module.phoneme_encoder.src_word_emb(batch[3]).detach()

                    D_t_real = model.module.D_t(
                        model.module.variance_adaptor.upsample,
                        texts_emb_d, real_mel,
                        max_real_mel_len, real_mel_masks, real_durations,
                    )
                    D_t_fake_for_d = model.module.D_t(
                        model.module.variance_adaptor.upsample,
                        texts_emb_d, gen_mel.detach(),
                        max_gen_mel_len, gen_mel_masks, gen_durations.detach(),
                    )
                    d_t_loss = 0.5 * (
                        F.mse_loss(D_t_real, torch.ones_like(D_t_real))
                        + F.mse_loss(D_t_fake_for_d, torch.zeros_like(D_t_fake_for_d))
                    )
                    backward(model, optimizer_disc, d_t_loss, step, grad_acc_step, grad_clip_thresh)

                # Meta-learning path (unused in the default recipe; kept for parity).
                if step > meta_learning_warmup:
                    output_disc = model.module.meta_learner_2(*(batch[2:]))
                    losses_2 = Loss_2(batch[2], output_disc)
                    total_loss_disc = losses_2[0]
                    backward(model, optimizer_disc, total_loss_disc, step, grad_acc_step, grad_clip_thresh)

                if step % log_step == 0:
                    if step > meta_learning_warmup:
                        losses = [l.item() for l in (losses_1+losses_2[1:])]
                    else:
                        losses = [l.item() for l in (losses_1+tuple([torch.zeros(1).to(device) for _ in range(3)]))]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    ce_e = emotion_ce_loss.item()   if emotion_ce_loss   is not None else 0.0
                    ce_i = intensity_ce_loss.item() if intensity_ce_loss is not None else 0.0
                    adv  = adv_g_loss.item()        if adv_g_loss        is not None else 0.0
                    message2 = ("Total: {:.4f}, Mel: {:.4f}, Pitch: {:.4f}, Energy: {:.4f}, "
                                "Duration: {:.4f}, Emotion-CE: {:.4f}, Intensity-CE: {:.4f}, "
                                "Adv: {:.4f}").format(
                        total_loss.item(), losses[1], losses[2], losses[3], losses[4],
                        ce_e, ce_i, adv)

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                    if wandb_run is not None:
                        wandb_payload = {
                            "train/total_loss":     losses[0],
                            "train/mel_loss":       losses[1],
                            "train/pitch_loss":     losses[2],
                            "train/energy_loss":    losses[3],
                            "train/duration_loss":  losses[4],
                        }
                        if emotion_ce_loss is not None:
                            wandb_payload["train/emotion_ce_loss"] = emotion_ce_loss.item()
                            wandb_payload["train/emotion_aux_acc"] = (
                                (aux_logits.argmax(dim=-1) == emotion_labels)
                                .float().mean().item()
                            )
                        if intensity_ce_loss is not None:
                            wandb_payload["train/intensity_ce_loss"] = intensity_ce_loss.item()
                            wandb_payload["train/intensity_aux_acc"] = (
                                (int_logits.argmax(dim=-1) == intensity_labels)
                                .float().mean().item()
                            )
                        if adv_g_loss is not None:
                            wandb_payload["train/adv_g_loss"]        = adv_g_loss.item()
                            wandb_payload["train/effective_lambda_adv"] = effective_lambda_adv
                        if d_t_loss is not None:
                            wandb_payload["train/d_t_loss"] = d_t_loss.item()
                        wandb.log(wandb_payload, step=step)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output[2:],
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                    if wandb_run is not None:
                        wandb.log({
                            "train/audio_reconstructed": wandb.Audio(
                                wav_reconstruction, sample_rate=sampling_rate,
                                caption=f"step_{step}_{tag}_recon"),
                            "train/audio_synthesized":   wandb.Audio(
                                wav_prediction, sample_rate=sampling_rate,
                                caption=f"step_{step}_{tag}_synth"),
                        }, step=step)

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, len(losses),
                                       combine_mode=args.combine_mode)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer_main": optimizer_main._optimizer.state_dict(),
                            "optimizer_disc": optimizer_disc._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--pretrained_ckpt", type=str, default=None,
        help="Warm-start from a foreign checkpoint. Loaded with strict=False; "
             "shape-mismatched or missing keys are skipped.",
    )
    parser.add_argument(
        "--lambda_emotion", type=float, default=0.1,
        help="Weight for the emotion-CE auxiliary loss on the style vector (0 disables).",
    )
    parser.add_argument(
        "--lambda_intensity", type=float, default=0.0,
        help="Weight for the intensity-CE auxiliary loss on the style vector (0 disables).",
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0.0,
        help="Weight for the D_t adversarial loss on generated mels (0 disables).",
    )
    parser.add_argument(
        "--adv_warmup_steps", type=int, default=1000,
        help="Linearly ramp lambda_adv from 0 to its full value over this many steps.",
    )
    parser.add_argument(
        "--combine_mode", type=str, default=None,
        choices=[None, "sum", "gated", "concat_linear"],
        help="Three-input factorization: combine the face style with the speaker "
             "channel into the style vector. 'concat_linear' is the default recipe; "
             "None falls back to face-only.",
    )
    parser.add_argument(
        "--speaker_emb", action="store_true",
        help="Use a pretrained ECAPA speaker embedding (cached in spk_emb_cache/) as "
             "the speaker channel. The default recipe. Requires --combine_mode.",
    )
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml",
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml",
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    # Expand $ENV_VARS and ~ in any path field.
    for cfg in (preprocess_config, train_config):
        for k, v in list(cfg.get("path", {}).items()):
            if isinstance(v, str):
                cfg["path"][k] = os.path.expandvars(os.path.expanduser(v))

    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
