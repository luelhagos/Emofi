"""MEAD-specific preprocessing helpers.

Source layout: <corpus_path>/<utt>.wav with <utt>_CorrectedTranscript.txt (or
<utt>.txt), where <utt> = <speaker>_<emotion>_<level>_<index>.

Pipeline:
    1. prepare_align(config): stage <raw_path>/<speaker>/<utt>.wav + <utt>.lab
       for the Montreal Forced Aligner, which writes <mfa_out>/<speaker>/<utt>.TextGrid.
    2. nest_textgrids(<mfa_out>, <preprocessed_path>/TextGrid): reorganize the
       TextGrids as <speaker>/<emotion>/<utt>.TextGrid, the layout preprocess.py expects.
    3. python3 preprocess.py config/MEAD/preprocess.yaml
"""

import os
import shutil

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def _speaker_from(basename: str) -> str:
    """'M027_angry_level1_024' -> 'M027'"""
    return basename.split("_")[0]


def _emotion_from(basename: str) -> str:
    """'M027_angry_level1_024' -> 'angry'"""
    return basename.split("_")[1]


def _transcript_path(corpus_path, basename):
    """Prefer *_CorrectedTranscript.txt; fall back to <basename>.txt."""
    for suffix in ("_CorrectedTranscript.txt", ".txt"):
        candidate = os.path.join(corpus_path, basename + suffix)
        if os.path.exists(candidate):
            return candidate
    return None



def prepare_align(config):
    in_dir        = config["path"]["corpus_path"]
    out_dir       = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners      = config["preprocessing"]["text"]["text_cleaners"]

    wav_names = sorted(f for f in os.listdir(in_dir) if f.endswith(".wav"))
    print(f"[mead.prepare_align] {len(wav_names)} wavs in {in_dir}")

    skipped_no_text = 0
    skipped_empty   = 0
    written         = 0

    for wav_name in tqdm(wav_names):
        basename = wav_name[:-4]
        speaker  = _speaker_from(basename)

        text_path = _transcript_path(in_dir, basename)
        if text_path is None:
            skipped_no_text += 1
            continue
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read().strip().replace("\n", " ")
        if not raw_text:
            skipped_empty += 1
            continue
        text = _clean_text(raw_text, cleaners)

        wav, _ = librosa.load(os.path.join(in_dir, wav_name), sr=sampling_rate)
        peak = float(np.max(np.abs(wav))) if wav.size else 0.0
        if peak == 0.0:
            skipped_empty += 1
            continue
        wav = wav / peak * max_wav_value

        spk_dir = os.path.join(out_dir, speaker)
        os.makedirs(spk_dir, exist_ok=True)
        wavfile.write(
            os.path.join(spk_dir, basename + ".wav"),
            sampling_rate,
            wav.astype(np.int16),
        )
        with open(os.path.join(spk_dir, basename + ".lab"), "w", encoding="utf-8") as f:
            f.write(text)
        written += 1

    print(
        f"[mead.prepare_align] wrote {written}  "
        f"skipped {skipped_no_text} (no transcript)  "
        f"{skipped_empty} (empty/silent)"
    )



def nest_textgrids(flat_root: str, nested_root: str, copy: bool = False) -> None:
    """Reorganize MFA output: <flat>/<spk>/<utt>.TextGrid -> <nested>/<spk>/<emo>/<utt>.TextGrid.

    Default is to move (rename); pass copy=True to preserve the originals.
    """
    if not os.path.isdir(flat_root):
        raise FileNotFoundError(f"MFA TextGrid root not found: {flat_root}")
    os.makedirs(nested_root, exist_ok=True)

    moved = 0
    for speaker in sorted(os.listdir(flat_root)):
        spk_in = os.path.join(flat_root, speaker)
        if not os.path.isdir(spk_in):
            continue
        for tg_name in os.listdir(spk_in):
            if not tg_name.endswith(".TextGrid"):
                continue
            basename = tg_name[:-len(".TextGrid")]
            emotion  = _emotion_from(basename)
            dst_dir  = os.path.join(nested_root, speaker, emotion)
            os.makedirs(dst_dir, exist_ok=True)
            src = os.path.join(spk_in, tg_name)
            dst = os.path.join(dst_dir, tg_name)
            if copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            moved += 1
    print(f"[mead.nest_textgrids] {'copied' if copy else 'moved'} {moved} TextGrids")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MEAD-specific preprocessing helpers.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_nest = sub.add_parser("nest_textgrids", help="Reorganise MFA output by emotion.")
    p_nest.add_argument("--flat",   required=True, help="MFA output dir (flat <spk>/<utt>.TextGrid).")
    p_nest.add_argument("--nested", required=True, help="Destination root (e.g. preprocessed_data/MEAD/TextGrid).")
    p_nest.add_argument("--copy",   action="store_true", help="Copy instead of move.")
    args = parser.parse_args()

    if args.cmd == "nest_textgrids":
        nest_textgrids(args.flat, args.nested, copy=args.copy)

