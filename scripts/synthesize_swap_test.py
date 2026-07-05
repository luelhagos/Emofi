"""
Cross-identity swap test for the three-input factorization.

Each test utterance is synthesized with:
  - face_embs  = this utterance's face     (intended emotion source)
  - reference  = a different speaker's     (intended speaker source)

The output filename encodes both the face emotion and the reference emotion, so
the SER evaluation can score the output two ways:
  - against the FACE emotion: high accuracy means the face drives emotion
  - against the REFERENCE emotion: high accuracy means the reference drives it

Two symlink trees are written for the SER tool (which parses
<spk>_<emotion>_<level>.wav):
  <out_dir>_byface/  labelled by the FACE emotion
  <out_dir>_byref/   labelled by the REFERENCE emotion
Run the SER evaluation on each and compare which label set the output matches.

Usage:
    python3 scripts/synthesize_swap_test.py \
        --restore_step 30000 \
        -p config/MEAD/preprocess.yaml -m config/MEAD/model.yaml \
        -t config/MEAD/train.yaml \
        --combine_mode concat_linear --speaker_emb \
        --out_dir eval/swap
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from tqdm import tqdm

sys.path.insert(0, ".")
from text import text_to_sequence
from utils.model import get_model, get_vocoder
from utils.tools import pad_1D, pad_2D

EMOTIONS = ["angry", "fear", "happy", "surprised", "sad", "neutral"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--restore_step", type=int, required=True)
    p.add_argument("-p", "--preprocess_config", required=True)
    p.add_argument("-m", "--model_config",      required=True)
    p.add_argument("-t", "--train_config",      required=True)
    p.add_argument("--manifest", default="test.txt")
    p.add_argument("--out_dir",  required=True,
                   help="Base dir. Two sibling _byface/_byref symlink trees are "
                        "created next to it for SER scoring.")
    p.add_argument("--combine_mode", type=str, required=True,
                   choices=["sum", "gated", "concat_linear"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="Controls the random pairing of face<==>ref. Fixed for reproducibility.")
    p.add_argument("--speaker_emb", action="store_true",
                   help="Speaker source is a DIFFERENT speaker's cached ECAPA emb "
                        "(spk_emb_cache/) instead of a ref mel. ECAPA is emotion-invariant "
                        "so the ref need only differ in SPEAKER. Must match how the model "
                        "was trained.")
    p.add_argument("--postfilter_refine", action="store_true",
                   help="Apply the intensity postfilter before vocoding. The "
                        "postfilter is style-conditioned, so speaker transfer "
                        "must be measured with it applied, not on the bare "
                        "backbone.")
    p.add_argument("--postfilter_levels", type=str, default=None,
                   help="Comma-separated levels the postfilter is applied to "
                        "(reference recipe: 'level2,level3'); others pass "
                        "through. Matches synthesize_test_set.py.")
    p.add_argument("--predict_level", action="store_true",
                   help="Predict the level from the face (intensity aux head) "
                        "instead of the filename -- the image-only condition.")
    p.add_argument("--spk_emb_dir", default=None,
                   help="Override the speaker-embedding cache directory (default "
                        "<preprocessed>/spk_emb_cache), e.g. to evaluate with a "
                        "post-processed embedding cache. The same --seed produces "
                        "the same pairs, enabling paired comparisons.")
    return p.parse_args()


def expand_path_vars(cfg):
    for k, v in list(cfg.get("path", {}).items()):
        if isinstance(v, str):
            cfg["path"][k] = os.path.expandvars(os.path.expanduser(v))


def main():
    args = parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)

    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config      = yaml.load(open(args.model_config),      Loader=yaml.FullLoader)
    train_config      = yaml.load(open(args.train_config),      Loader=yaml.FullLoader)
    expand_path_vars(preprocess_config); expand_path_vars(train_config)
    configs = (preprocess_config, model_config, train_config)

    args.out_dir = os.path.expandvars(os.path.expanduser(args.out_dir))
    os.makedirs(args.out_dir, exist_ok=True)
    byface = args.out_dir + "_byface"; os.makedirs(byface, exist_ok=True)
    byref  = args.out_dir + "_byref";  os.makedirs(byref,  exist_ok=True)

    model   = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    pre = preprocess_config["path"]["preprocessed_path"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    face_dir = os.path.join(pre, "face_emb_cache")
    spk_dir  = os.path.join(pre, "spk_emb_cache")
    if args.spk_emb_dir:
        spk_dir = os.path.expandvars(os.path.expanduser(args.spk_emb_dir))
        print(f"[swap] ECAPA cache OVERRIDE: {spk_dir}")

    rows = []
    with open(os.path.join(pre, args.manifest)) as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) != 4:
                continue
            base, spk, phones, _ = parts
            emo = base.split("_")[1]
            rows.append({"base": base, "spk": spk, "phones": phones, "emo": emo})

    # Index utterances by emotion + speaker, to pick references that come from
    # a different speaker AND a different emotion than the face.
    by_emo = {e: [] for e in EMOTIONS}
    for r in rows:
        if r["emo"] in by_emo:
            by_emo[r["emo"]].append(r)

    def pick_ref(face_row):
        """A ref from a different speaker AND different emotion than the face."""
        other_emos = [e for e in EMOTIONS if e != face_row["emo"] and by_emo[e]]
        rng.shuffle(other_emos)
        for e in other_emos:
            cands = [r for r in by_emo[e] if r["spk"] != face_row["spk"]]
            if cands:
                return rng.choice(cands)
        return None

    written = skipped = errored = 0
    pairs_log = []

    for r in tqdm(rows):
        base, spk, phones, face_emo = r["base"], r["spk"], r["phones"], r["emo"]
        ref = pick_ref(r)
        if ref is None:
            skipped += 1; continue

        face_path = os.path.join(face_dir, f"{base}.pt")
        mel_path  = os.path.join(pre, "mel", f"{spk}-mel-{base}.npy")
        if args.speaker_emb:
            ref_src_path = os.path.join(spk_dir, f"{ref['base']}.pt")     # ECAPA emb
        else:
            ref_src_path = os.path.join(pre, "mel", f"{ref['spk']}-mel-{ref['base']}.npy")
        if not (os.path.exists(face_path) and os.path.exists(mel_path)
                and os.path.exists(ref_src_path)):
            skipped += 1; continue

        # Absolute so the _byface/_byref symlinks resolve regardless of CWD
        # or whether --out_dir was passed relative.
        out_wav = os.path.abspath(os.path.join(args.out_dir, f"{base}.wav"))
        try:
            face_embs = torch.load(face_path, map_location="cpu").float().unsqueeze(0).to(device)
            mel       = np.load(mel_path)                 # own mel (for length only)

            seq = np.array(text_to_sequence(phones, cleaners))
            texts = torch.from_numpy(pad_1D([seq])).long().to(device)
            src_lens = torch.from_numpy(np.array([len(seq)])).to(device)
            mels = torch.from_numpy(pad_2D([mel])).float().to(device)
            mel_lens = torch.from_numpy(np.array([mel.shape[0]])).to(device)

            ref_mels = spk_embs = None
            if args.speaker_emb:
                spk_embs = torch.load(ref_src_path, map_location="cpu").float().unsqueeze(0).to(device)
            else:
                ref_mel  = np.load(ref_src_path)          # SWAP: different speaker+emotion mel
                ref_mels = torch.from_numpy(pad_2D([ref_mel])).float().to(device)

            with torch.no_grad():
                output = model(
                    None, texts, src_lens, int(len(seq)),
                    mels, mel_lens, int(mel.shape[0]),
                    None, None, None,
                    face_embs=face_embs,
                    ref_mels=ref_mels,
                    spk_embs=spk_embs,
                    combine_mode=args.combine_mode,
                )
                mel_out = output[0]
                # Intensity postfilter, gated and level-conditioned exactly as in
                # synthesize_test_set.py, so transfer is measured on the deployed
                # system.
                lvl = base.split("_")[2]
                if args.predict_level and getattr(model, "intensity_aux_head",
                                                  None) is not None:
                    pred = int(model.compute_intensity_aux(face_embs)
                               .argmax(-1).item())
                    lvl = ("level1", "level2", "level3")[pred]
                pf_levels = (set(args.postfilter_levels.split(","))
                             if args.postfilter_levels else None)
                if args.postfilter_refine and (pf_levels is None or lvl in pf_levels):
                    if getattr(model, "postfilter_g", None) is None:
                        raise SystemExit("--postfilter_refine needs a checkpoint "
                                         "with model config postfilter.enabled.")
                    style_vec = model.compute_style_vector(
                        face_embs, spk_embs=spk_embs, ref_mels=ref_mels,
                        combine_mode=args.combine_mode)
                    lvl_idx = torch.tensor(
                        [{"level1": 0, "level2": 1, "level3": 2}.get(lvl, 2)],
                        device=device)
                    mel_out = model.postfilter_refine(mel_out, style_vec, lvl_idx)
                wav = vocoder(mel_out.transpose(1, 2)).squeeze().cpu().numpy()

            wav = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
            wavfile.write(out_wav, sr, wav)

            # SER-parseable symlinks: <spk><idx>_<emotion>_<level>.wav
            idx = base.split("_")[3]; lvl = base.split("_")[2]
            face_name = f"{spk}{idx}_{face_emo}_{lvl}.wav"
            ref_name  = f"{spk}{idx}_{ref['emo']}_{lvl}.wav"
            for d, nm in ((byface, face_name), (byref, ref_name)):
                ln = os.path.join(d, nm)
                if not os.path.exists(ln):
                    os.symlink(out_wav, ln)

            pairs_log.append(f"{base}|face={face_emo}|ref={ref['base']}|ref_emo={ref['emo']}")
            written += 1
        except Exception as e:
            print(f"[err] {base}: {e}"); errored += 1

    with open(args.out_dir + "_pairs.txt", "w") as f:
        f.write("\n".join(pairs_log) + "\n")

    print(f"\nDone. wrote {written}, skipped {skipped}, errored {errored}")
    print(f"  audio:        {args.out_dir}")
    print(f"  SER by FACE:  {byface}   (high acc here ==> FACE drives emotion ==> decoupling works)")
    print(f"  SER by REF:   {byref}    (high acc here ==> REF drives emotion ==> factorization failed)")
    print(f"  pairing log:  {args.out_dir}_pairs.txt")


if __name__ == "__main__":
    main()

