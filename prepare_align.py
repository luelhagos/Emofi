"""Stage MEAD utterances for forced alignment.

Reads <corpus_path>/<utt>.wav and its transcript, writes
<raw_path>/<speaker>/<utt>.wav (resampled, peak-normalized) and <utt>.lab for
the Montreal Forced Aligner.

Usage:
    python3 prepare_align.py config/MEAD/preprocess.yaml
"""

import argparse

import yaml

from preprocessor import mead


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    mead.prepare_align(config)
