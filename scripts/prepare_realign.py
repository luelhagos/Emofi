"""
Transcript correction, stage 3: build the MFA re-alignment corpus.

Assigns each utterance its corrected (canonical) text from the two-source
recovery (match_canonical.py + whisper_verify.py), then decides per utterance
whether the existing TextGrid was aligned on the wrong text:
  KEEP     TextGrid words already match the corrected text (>= --keep_thresh)
  REALIGN  TextGrid words differ and the corrected text is confident
           -> copied into mfa_corpus/<speaker>/<base>.wav (+ .lab) for MFA
  EXCLUDE  no confident corrected text (disagreement / multi-sentence / none)

Confidence for the corrected text (conservative):
  content_hi   content-match ratio >= 0.90
  agree        content + Whisper map to the same canonical
  whisper      Whisper maps to a canonical at >= 0.85
  content_mid  content-match ratio >= 0.80
  else exclude; multi-sentence clips (--multi_ratio) are excluded.

Outputs (--out_dir):
  corrected_transcripts.tsv   base, status, conf, corrected_text
  keep.txt / realign.txt / exclude.txt
  mfa_corpus/<speaker>/<base>.wav (+ .lab)   REALIGN only

Usage:
  python3 scripts/prepare_realign.py \
      --pre data/MEAD \
      --raw data/MEAD_processed \
      --match_tsv eval/transcripts_canonical/match_canonical.tsv \
      --whisper_tsv eval/transcripts_canonical/whisper_verify.tsv \
      --out_dir eval/realign_prep
"""

import argparse
import csv
import difflib
import os
import re
import shutil

import tgt


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def textgrid_words(path):
    tg = tgt.io.read_textgrid(path)
    try:
        tier = tg.get_tier_by_name("words")
    except Exception:
        return None
    return [w for w in (((itv.text or "").strip().lower()) for itv in tier._objects)
            if w and w not in ("sil", "sp", "spn", "<unk>")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--match_tsv", required=True)
    p.add_argument("--whisper_tsv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tg_dir", default=None, help="default <pre>/TextGrid")
    p.add_argument("--keep_thresh", type=float, default=0.85,
                   help="TextGrid-vs-corrected similarity at/above which the utterance is kept (no re-align)")
    p.add_argument("--multi_ratio", type=float, default=1.4,
                   help="whisper_words / canonical_words above this = multi-sentence -> exclude")
    return p.parse_args()


def main():
    a = parse_args()
    for k in ("pre", "raw", "match_tsv", "whisper_tsv", "out_dir"):
        setattr(a, k, os.path.expandvars(os.path.expanduser(getattr(a, k))))
    tg_dir = os.path.expandvars(os.path.expanduser(a.tg_dir or os.path.join(a.pre, "TextGrid")))
    os.makedirs(a.out_dir, exist_ok=True)

    content = {}
    with open(a.match_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            content[r["base"]] = (r["canonical_text"], float(r["ratio"]), r["tier"])

    whisper = {}
    with open(a.whisper_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            whisper[r["base"]] = (r.get("whisper_text", ""), r.get("whisper_canonical", ""),
                                  float(r.get("whisper_ratio", 0) or 0))

    # 1. assign corrected text + confidence
    def assign(base):
        c_text, c_ratio, c_tier = content[base]
        if c_tier == "hi":
            return c_text, "content_hi"
        if base in whisper:
            wtext, wcanon, wratio = whisper[base]
            ref = wcanon or c_text
            if ref and len(wtext.split()) > a.multi_ratio * max(1, len(ref.split())):
                return None, "exclude_multi"
            if wcanon and norm(wcanon) == norm(c_text) and c_ratio >= 0.6:
                return c_text, "agree"
            if wratio >= 0.85 and wcanon:
                return wcanon, "whisper"
            if c_ratio >= 0.80:
                return c_text, "content_mid"
            return None, "exclude_disagree"
        return None, "exclude_no_whisper"

    # 2. classify keep / realign / exclude against the existing TextGrid
    tg_idx = {}  # basename -> path, indexed once
    for root, _, files in os.walk(tg_dir):
        for fn in files:
            if fn.endswith(".TextGrid"):
                tg_idx[fn[:-9]] = os.path.join(root, fn)
    print(f"[prepare] indexed {len(tg_idx)} TextGrids")

    rows = []
    counts = {"keep": 0, "realign": 0, "exclude_multi": 0, "exclude_disagree": 0,
              "exclude_no_whisper": 0, "exclude_no_tg": 0}
    realign_bases = []
    for base in content:
        corrected, conf = assign(base)
        if corrected is None:
            counts[conf] = counts.get(conf, 0) + 1
            rows.append((base, "exclude", conf, ""))
            continue
        tgp = tg_idx.get(base)
        if not tgp:
            counts["exclude_no_tg"] += 1
            rows.append((base, "exclude", "no_textgrid", corrected))
            continue
        tgw = textgrid_words(tgp) or []
        sim = difflib.SequenceMatcher(None, norm(" ".join(tgw)), norm(corrected)).ratio()
        if sim >= a.keep_thresh:
            counts["keep"] += 1
            rows.append((base, "keep", conf, corrected))
        else:
            counts["realign"] += 1
            realign_bases.append(base)
            rows.append((base, "realign", conf, corrected))

    # 3. write tsv + lists
    with open(os.path.join(a.out_dir, "corrected_transcripts.tsv"), "w") as f:
        f.write("base\tstatus\tconf\tcorrected_text\n")
        for b, st, cf, txt in rows:
            f.write(f"{b}\t{st}\t{cf}\t{txt}\n")
    for name, st in (("keep.txt", "keep"), ("realign.txt", "realign")):
        with open(os.path.join(a.out_dir, name), "w") as f:
            for b, s, _, _ in rows:
                if s == st:
                    f.write(b + "\n")
    with open(os.path.join(a.out_dir, "exclude.txt"), "w") as f:
        for b, s, cf, _ in rows:
            if s == "exclude":
                f.write(f"{b}\t{cf}\n")

    # 4. build the MFA corpus for the realign set (copy wav + write .lab)
    raw_idx = {}
    for root, _, files in os.walk(a.raw):
        for fn in files:
            if fn.endswith(".wav"):
                raw_idx[fn[:-4]] = os.path.join(root, fn)
    corp = os.path.join(a.out_dir, "mfa_corpus")
    os.makedirs(corp, exist_ok=True)
    corrected_by_base = {b: txt for b, st, _, txt in rows if st == "realign"}
    copied = miss = 0
    for base in realign_bases:
        wav = raw_idx.get(base)
        if not wav:
            miss += 1
            continue
        spk = base.split("_")[0]
        d = os.path.join(corp, spk)
        os.makedirs(d, exist_ok=True)
        shutil.copy(wav, os.path.join(d, base + ".wav"))
        with open(os.path.join(d, base + ".lab"), "w") as f:
            f.write(corrected_by_base[base].strip() + "\n")
        copied += 1

    # 5. report
    tot = len(content)
    print(f"=== prepare_realign over {tot} utterances ===")
    for k in ("keep", "realign", "exclude_multi", "exclude_disagree",
              "exclude_no_whisper", "exclude_no_tg"):
        v = counts.get(k, 0)
        print(f"  {k:18s}: {v:6d} ({v/tot:.2%})")
    print(f"\n  MFA corpus: copied {copied} wav+lab pairs ({miss} missing wav) -> {corp}")
    print(f"  corrected text for KEEP+REALIGN -> corrected_transcripts.tsv "
          f"(EXCLUDE rows are dropped from the rebuilt manifests).")


if __name__ == "__main__":
    main()
