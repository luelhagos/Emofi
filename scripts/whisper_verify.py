"""
Transcript correction, stage 2: Whisper as an independent second source plus
an audio-completeness check.

For the utterances stage 1 (match_canonical.py) could not confidently assign
(whisper_needed.txt), and/or every base in a manifest (--manifest), transcribe
the raw audio with Whisper, match the transcript to the canonical pool, and
compare to the stage-1 assignment. Per utterance:
  agree       Whisper's canonical match == stage-1 content match
  audio_note  'audio_complete' if Whisper recovers a full canonical sentence
              (ratio >= --agree_thresh), else 'whisper_also_short'

Output: <out_txt> TSV with base, whisper_text, whisper_canonical,
whisper_ratio, content_canonical, agree, audio_note, plus a printed summary.

Usage:
  python3 scripts/whisper_verify.py \
      --raw data/MEAD_processed \
      --corpus data/mead_sentences.csv \
      --match_tsv eval/transcripts_canonical/match_canonical.tsv \
      --bases eval/transcripts_canonical/whisper_needed.txt \
      --model large --out_txt eval/transcripts_canonical/whisper_verify.tsv

Requires: pip install openai-whisper
"""

import argparse
import csv
import difflib
import os
import re

import whisper


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


def index_raw(raw):
    idx = {}
    for root, _, files in os.walk(raw):
        for fn in files:
            if fn.endswith(".wav"):
                idx[fn[:-4]] = os.path.join(root, fn)
    return idx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="raw wav dir (nested ok)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--match_tsv", required=True, help="stage-1 match_canonical.tsv")
    p.add_argument("--bases", help="file of basenames to verify (one per line)")
    p.add_argument("--manifest", help="alternatively, verify every base in this manifest")
    p.add_argument("--model", default="large")
    p.add_argument("--language", default="en")
    p.add_argument("--agree_thresh", type=float, default=0.90,
                   help="ratio at/above which a Whisper->canonical match counts as a full recovery")
    p.add_argument("--out_txt", required=True)
    p.add_argument("--max_files", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    for k in ("raw", "corpus", "match_tsv", "out_txt"):
        setattr(a, k, os.path.expandvars(os.path.expanduser(getattr(a, k))))

    pool = load_corpus(a.corpus)

    content = {}
    with open(a.match_tsv, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            content[r["base"]] = (r["canonical_text"], float(r["ratio"]), r["tier"])

    bases = []
    if a.bases:
        for line in open(os.path.expandvars(os.path.expanduser(a.bases))):
            if line.strip():
                bases.append(line.strip())
    if a.manifest:
        for line in open(os.path.expandvars(os.path.expanduser(a.manifest)), encoding="utf-8"):
            p = line.rstrip("\n").split("|")
            if p and p[0]:
                bases.append(p[0])
    bases = list(dict.fromkeys(bases))  # de-dup, keep order
    if a.max_files:
        bases = bases[:a.max_files]
    print(f"[verify] {len(bases)} utterances to transcribe")

    raw_idx = index_raw(a.raw)
    print(f"[verify] indexed {len(raw_idx)} raw wavs; loading Whisper '{a.model}' ...")
    model = whisper.load_model(a.model)

    rows = []
    scored = agree = audio_ok = 0
    for i, b in enumerate(bases):
        wp = raw_idx.get(b)
        if not wp:
            rows.append((b, "", "", 0.0, "", "NA", "no_raw_wav"))
            continue
        try:
            wt = norm(model.transcribe(wp, language=a.language, fp16=False)["text"])
        except Exception as e:
            rows.append((b, "ERR", "", 0.0, "", "NA", str(e)[:40]))
            continue
        wr, wc = best_match(wt, pool)
        cc = content.get(b, ("",))[0]
        ag = 1 if (wc and cc and norm(wc) == norm(cc)) else 0
        note = "audio_complete" if wr >= a.agree_thresh else "whisper_also_short"
        rows.append((b, wt, wc, f"{wr:.3f}", cc, ag, note))
        scored += 1
        agree += ag
        audio_ok += 1 if wr >= a.agree_thresh else 0
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(bases)}")

    with open(a.out_txt, "w") as f:
        f.write("base\twhisper_text\twhisper_canonical\twhisper_ratio\tcontent_canonical\tagree\taudio_note\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")

    print(f"\n=== Whisper verify ({scored} scored) ===")
    if scored:
        print(f"  agree with stage-1 content match : {agree}/{scored} = {agree/scored:.1%}")
        print(f"  audio complete (full sentence)   : {audio_ok}/{scored} = {audio_ok/scored:.1%}")
        print(f"  -> 'whisper_also_short' rows are the physically clipped suspects")
    print(f"  report -> {a.out_txt}")


if __name__ == "__main__":
    main()
