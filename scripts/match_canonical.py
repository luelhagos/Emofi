"""
Transcript correction, stage 1: content-match each utterance's transcript to
the canonical MEAD sentence pool (mead_sentences.csv, column `sentence`).

Each utterance is assigned by fuzzy content match (difflib ratio) against the
full pool. Tiers by best ratio:
  hi  (>= --hi, default 0.90): confident -> safe to assign
  mid (>= --lo): recoverable, verify with Whisper in stage 2 (whisper_verify.py)
  lo  (<  --lo): garbled beyond fuzzy match -> Whisper / manual / exclude

Outputs (under --out_dir):
  match_canonical.tsv  base, split, ratio, tier, old_text, canonical_text
  whisper_needed.txt   bases in mid/lo tiers (the stage-2 work list)
With --write_manifests, writes corrected train/val/test using only hi-tier
matches; mid/lo keep their original text. Phones/durations are unchanged.

Usage:
  python3 scripts/match_canonical.py \
      --pre data/MEAD \
      --corpus data/mead_sentences.csv \
      --out_dir eval/transcripts_canonical
"""

import argparse
import csv
import difflib
import os
import re


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def load_corpus(path):
    pool, seen = [], set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n = norm(r["sentence"])
            if n and n not in seen:
                seen.add(n)
                pool.append((n, r["sentence"]))
    return pool


def best_match(text, pool):
    best = (0.0, None)
    for n, raw in pool:
        ratio = difflib.SequenceMatcher(None, text, n).ratio()
        if ratio > best[0]:
            best = (ratio, raw)
    return best


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True, help="dir with train/val/test.txt")
    p.add_argument("--corpus", required=True, help="mead_sentences.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--hi", type=float, default=0.90)
    p.add_argument("--lo", type=float, default=0.60)
    p.add_argument("--write_manifests", action="store_true",
                   help="write hi-tier-corrected manifests (mid/lo unchanged)")
    return p.parse_args()


def main():
    a = parse_args()
    a.pre = os.path.expandvars(os.path.expanduser(a.pre))
    a.corpus = os.path.expandvars(os.path.expanduser(a.corpus))
    a.out_dir = os.path.expandvars(os.path.expanduser(a.out_dir))
    os.makedirs(a.out_dir, exist_ok=True)

    pool = load_corpus(a.corpus)
    print(f"[corpus] {len(pool)} distinct canonical sentences")

    rows = []
    for s in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(a.pre, s)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            parts = line.rstrip("\n").split("|")
            if len(parts) == 4:
                rows.append((parts[0], s, parts[3]))

    cache = {}  # normalized text -> (ratio, canonical); many utts share text
    tiers = {s: {"hi": 0, "mid": 0, "lo": 0} for s in ("train.txt", "val.txt", "test.txt")}
    assign = {}  # base -> canonical, hi-tier only

    out = open(os.path.join(a.out_dir, "match_canonical.tsv"), "w")
    out.write("base\tsplit\tratio\ttier\told_text\tcanonical_text\n")
    need = open(os.path.join(a.out_dir, "whisper_needed.txt"), "w")
    for i, (base, s, old) in enumerate(rows):
        nt = norm(old)
        if nt not in cache:
            cache[nt] = best_match(nt, pool)
        ratio, canon = cache[nt]
        tier = "hi" if ratio >= a.hi else ("mid" if ratio >= a.lo else "lo")
        tiers[s][tier] += 1
        out.write(f"{base}\t{s}\t{ratio:.3f}\t{tier}\t{old}\t{canon}\n")
        if tier == "hi":
            assign[base] = canon
        else:
            need.write(base + "\n")
        if (i + 1) % 4000 == 0:
            print(f"  {i+1}/{len(rows)}")
    out.close()
    need.close()

    print("\n=== tier distribution ===")
    for s in tiers:
        t = tiers[s]
        n = sum(t.values())
        if n:
            print(f"  {s:10s}: hi={t['hi']/n:.1%}  mid={t['mid']/n:.1%}  lo={t['lo']/n:.1%}  (n={n})")
    nw = sum(t["mid"] + t["lo"] for t in tiers.values())
    print(f"\n  whisper_needed (mid+lo): {nw}")
    tt = tiers["test.txt"]
    if sum(tt.values()):
        print(f"  test hi-rate: {tt['hi']/sum(tt.values()):.1%}")

    if a.write_manifests:
        for s in ("train.txt", "val.txt", "test.txt"):
            p = os.path.join(a.pre, s)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fin, open(os.path.join(a.out_dir, s), "w") as fo:
                for line in fin:
                    parts = line.rstrip("\n").split("|")
                    if len(parts) == 4 and parts[0] in assign:
                        parts[3] = assign[parts[0]]
                    fo.write("|".join(parts) + "\n")
        print(f"  wrote hi-tier-corrected manifests -> {a.out_dir}/ (mid/lo pending stage 2)")


if __name__ == "__main__":
    main()
