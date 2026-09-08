"""Speaker similarity between synthesized and real audio (Resemblyzer d-vectors).

Paired condition (--synth_dirs): for each synthesized wav, cosine similarity to
the real recording of the same utterance (same basename in --real_dir), reported
per directory as mean, 95% CI, and n. Real-audio anchors are computed from
--real_dir alone: same-speaker pairs (two recordings of one speaker) and
different-speaker pairs, sampled with a fixed seed.

Swap condition (--swap_dir + --pairs_file): each output was synthesized from one
speaker's face and another speaker's reference, so there is no same-utterance
recording in the target voice; the score is the cosine to the reference
speaker's real-clip centroid, with the face speaker's centroid as the contrast.
The pairs file is the <out_dir>_pairs.txt written by synthesize_swap_test.py.

Embeddings are cached per directory under --cache_dir.

Usage:
    python3 scripts/eval_speaker_similarity.py \
        --real_dir data/MEAD_processed --synth_dirs <out_dir> [<out_dir2> ...] \
        --out_txt eval/speaker_similarity.txt

    python3 scripts/eval_speaker_similarity.py \
        --real_dir data/MEAD_processed --swap_dir <swap_dir> \
        --pairs_file <swap_dir>_pairs.txt --out_txt eval/speaker_similarity_swap.txt

Requires: pip install resemblyzer
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np


def speaker_of(base):
    return base.split("_")[0]


def load_encoder(device):
    from resemblyzer import VoiceEncoder, preprocess_wav
    enc = VoiceEncoder(device=device)

    def embed(path):
        return enc.embed_utterance(preprocess_wav(path))

    return embed


def embed_dir(wav_dir, embed, cache_dir, max_files=None):
    """Return {basename: unit-norm d-vector} for every wav in wav_dir (cached)."""
    tag = os.path.basename(os.path.normpath(wav_dir))
    cache = os.path.join(cache_dir, f"{tag}_resemblyzer.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["names"], z["embs"]))
    names = sorted(f[:-4] for f in os.listdir(wav_dir) if f.endswith(".wav"))
    if max_files:
        names = names[:max_files]
    embs = []
    for i, b in enumerate(names):
        e = np.asarray(embed(os.path.join(wav_dir, b + ".wav")), dtype=np.float32)
        embs.append(e / (np.linalg.norm(e) + 1e-8))
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i + 1}/{len(names)}")
    embs = np.stack(embs) if embs else np.zeros((0, 256), dtype=np.float32)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache, names=np.array(names), embs=embs)
    return dict(zip(names, embs))


def summarize(vals):
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return float(v.mean()), float(ci), int(len(v))


def fmt(label, vals):
    m, ci, n = summarize(vals)
    return f"{label:44s} {m:7.4f} +/- {ci:6.4f}   (n={n})"


def real_anchors(real, rng, n_pairs):
    """Real-audio reference values over random samples: same-speaker and
    different-speaker utterance pairs, and each utterance against its own
    speaker's centroid (leave-one-out) and another speaker's centroid."""
    by_spk = defaultdict(list)
    for b in real:
        by_spk[speaker_of(b)].append(b)
    speakers = [s for s in by_spk if len(by_spk[s]) > 1]
    sums = {s: np.sum([real[b] for b in by_spk[s]], axis=0) for s in speakers}
    cent = {s: sums[s] / (np.linalg.norm(sums[s]) + 1e-8) for s in speakers}
    same, diff, own_c, other_c = [], [], [], []
    for _ in range(n_pairs):
        s = rng.choice(speakers)
        a, b = rng.sample(by_spk[s], 2)
        same.append(float(real[a] @ real[b]))
        t = rng.choice([x for x in speakers if x != s])
        c = rng.choice(by_spk[t])
        diff.append(float(real[a] @ real[c]))
        loo = sums[s] - real[a]
        own_c.append(float(real[a] @ (loo / (np.linalg.norm(loo) + 1e-8))))
        other_c.append(float(real[a] @ cent[t]))
    return same, diff, own_c, other_c


def centroids(real):
    acc = defaultdict(list)
    for b, e in real.items():
        acc[speaker_of(b)].append(e)
    out = {}
    for s, es in acc.items():
        c = np.mean(es, axis=0)
        out[s] = c / (np.linalg.norm(c) + 1e-8)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", required=True, help="real wavs, <basename>.wav")
    p.add_argument("--synth_dirs", nargs="*", default=[],
                   help="synthesized dirs sharing basenames with --real_dir (paired condition)")
    p.add_argument("--swap_dir", default=None, help="swap-test outputs (swap condition)")
    p.add_argument("--pairs_file", default=None,
                   help="<swap_dir>_pairs.txt from synthesize_swap_test.py")
    p.add_argument("--cache_dir", default="eval/spk_sim_cache")
    p.add_argument("--n_anchor_pairs", type=int, default=2000)
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_txt", default=None)
    args = p.parse_args()
    if args.swap_dir and not args.pairs_file:
        raise SystemExit("--swap_dir requires --pairs_file")

    embed = load_encoder(args.device)
    rng = random.Random(args.seed)
    lines = []

    print("[spk-sim] embedding real audio ...")
    real = embed_dir(args.real_dir, embed, args.cache_dir, args.max_files)
    same, diff, own_c, other_c = real_anchors(real, rng, args.n_anchor_pairs)
    lines.append("Real-audio anchors (cosine, Resemblyzer d-vectors)")
    lines.append(fmt("  same speaker, different utterance", same))
    lines.append(fmt("  different speaker", diff))
    lines.append(fmt("  utterance vs own-speaker centroid", own_c))
    lines.append(fmt("  utterance vs other-speaker centroid", other_c))

    if args.synth_dirs:
        lines.append("")
        lines.append("Paired condition: synthesized vs the real recording of the same utterance")
        for d in args.synth_dirs:
            print(f"[spk-sim] embedding {d} ...")
            synth = embed_dir(d, embed, args.cache_dir, args.max_files)
            common = [b for b in synth if b in real]
            cos = [float(synth[b] @ real[b]) for b in common]
            lines.append(fmt(f"  {os.path.basename(os.path.normpath(d))}", cos))

    if args.swap_dir:
        print(f"[spk-sim] embedding {args.swap_dir} ...")
        swap = embed_dir(args.swap_dir, embed, args.cache_dir, args.max_files)
        cent = centroids(real)
        to_ref, to_face, ref_wins = [], [], 0
        with open(args.pairs_file) as f:
            for ln in f:
                parts = ln.strip().split("|")
                if len(parts) < 3 or parts[0] not in swap:
                    continue
                target = parts[0]
                ref_base = next(x.split("=", 1)[1] for x in parts if x.startswith("ref="))
                e = swap[target]
                r = float(e @ cent[speaker_of(ref_base)])
                fc = float(e @ cent[speaker_of(target)])
                to_ref.append(r)
                to_face.append(fc)
                ref_wins += int(r > fc)
        lines.append("")
        lines.append("Swap condition: synthesized vs speaker centroids of real clips")
        lines.append(fmt("  to the reference speaker (target voice)", to_ref))
        lines.append(fmt("  to the face speaker (should be lower)", to_face))
        n = len(to_ref)
        lines.append(f"  closer to the reference speaker: {ref_wins}/{n} "
                     f"({ref_wins / max(n, 1):.1%})")

    report = "\n".join(lines)
    print("\n" + report)
    if args.out_txt:
        os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
        with open(args.out_txt, "w") as f:
            f.write(report + "\n")
        print(f"\n[spk-sim] report -> {args.out_txt}")


if __name__ == "__main__":
    main()
