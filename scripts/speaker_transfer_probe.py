"""
Speaker-transfer probe for the cross-identity swap test.

Question: in a swap-synthesized utterance (face = emotion source, reference = a
different speaker), does the output voice match the reference speaker (speaker
transfer works) or the face's original speaker (leaked face identity
dominates)? This produces the speaker-transfer metric.

Requires: pip install speechbrain torchaudio  (or: pip install resemblyzer)

Method:
  1. Load a speaker-verification model (SpeechBrain ECAPA preferred,
     Resemblyzer fallback) -> maps a wav to a speaker embedding.
  2. Build per-speaker centroids from real MEAD audio (avg of N real
     utterances per speaker).
  3. Sanity baseline: classify real held-out MEAD utterances by nearest
     centroid. If this accuracy is low, the SV model cannot tell MEAD
     speakers apart and the probe is uninformative -- report and stop.
  4. For each swap output, compare cosine(emb, ref_centroid) vs
     cosine(emb, face_centroid). "Transfer success" = closer to ref.

Reads the swap pairing log written by synthesize_swap_test.py
(<out_dir>_pairs.txt) to know each output's face-speaker and ref-speaker.

Usage:
    python3 scripts/speaker_transfer_probe.py \
        --swap_dir   eval/swap \
        --pairs      eval/swap_pairs.txt \
        --real_dir   data/MEAD_processed \
        --centroid_n 10
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--swap_dir", required=True, help="Dir of swap-synth wavs (<base>.wav).")
    p.add_argument("--pairs",    required=True, help="<out_dir>_pairs.txt from the swap script.")
    p.add_argument("--real_dir", required=True, help="Real MEAD wavs (<base>.wav).")
    p.add_argument("--centroid_n", type=int, default=10,
                   help="Real utterances per speaker for the centroid.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_sv_backend(device):
    """Return (embed_fn, name). embed_fn: wav_path -> np.ndarray (D,)."""
    # Preferred: SpeechBrain ECAPA-TDNN (VoxCeleb).
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        def embed(path):
            import torchaudio
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            with torch.no_grad():
                e = clf.encode_batch(wav).squeeze().cpu().numpy()
            return e / (np.linalg.norm(e) + 1e-9)
        return embed, "speechbrain-ecapa"
    except Exception as e:
        print(f"[probe] SpeechBrain unavailable ({e}); trying Resemblyzer...")

    # Fallback: Resemblyzer.
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder(device=device)
        def embed(path):
            wav = preprocess_wav(path)
            e = enc.embed_utterance(wav)
            return e / (np.linalg.norm(e) + 1e-9)
        return embed, "resemblyzer"
    except Exception as e:
        raise SystemExit(f"[probe] No SV backend available. "
                         f"pip install speechbrain or resemblyzer. ({e})")


def cos(a, b):
    return float(np.dot(a, b))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    embed, backend = load_sv_backend(args.device)
    print(f"[probe] SV backend: {backend}")

    # gather real utterances per speaker (for centroids + sanity test)
    real_by_spk = defaultdict(list)
    for f in os.listdir(args.real_dir):
        if f.endswith(".wav"):
            spk = f.split("_")[0]
            real_by_spk[spk].append(os.path.join(args.real_dir, f))

    # centroid = mean of N real embeddings; hold out the rest for sanity test
    centroids, heldout = {}, {}
    for spk, files in real_by_spk.items():
        rng.shuffle(files)
        cset = files[:args.centroid_n]
        hset = files[args.centroid_n:args.centroid_n + 5]
        if len(cset) < 3:
            continue
        embs = [embed(p) for p in cset]
        centroids[spk] = np.mean(embs, axis=0)
        centroids[spk] /= (np.linalg.norm(centroids[spk]) + 1e-9)
        heldout[spk] = hset
    print(f"[probe] built {len(centroids)} speaker centroids "
          f"({args.centroid_n} utts each)")

    spk_list = sorted(centroids)

    # sanity check: nearest-centroid speaker classification on real held-out
    correct = total = 0
    for spk, files in heldout.items():
        for p in files:
            e = embed(p)
            pred = max(spk_list, key=lambda s: cos(e, centroids[s]))
            correct += int(pred == spk); total += 1
    sanity = correct / max(total, 1)
    print(f"\n[sanity] real-audio nearest-centroid speaker acc: {sanity:.3f} "
          f"({correct}/{total}) over {len(spk_list)} speakers "
          f"(chance = {1/len(spk_list):.3f})")
    if sanity < 0.5:
        print("[sanity] WARNING: SV model struggles to separate MEAD speakers; "
              "transfer numbers below are weak evidence.")

    # swap transfer test
    pairs = {}
    for line in open(args.pairs):
        line = line.strip()
        if not line: continue
        # base|face=<emo>|ref=<refbase>|ref_emo=<emo>
        parts = line.split("|")
        base = parts[0]
        ref_base = parts[2].split("=")[1]
        pairs[base] = {"face_spk": base.split("_")[0],
                       "ref_spk":  ref_base.split("_")[0]}

    ref_wins = face_wins = skipped = 0
    sim_to_ref, sim_to_face = [], []
    for base, meta in pairs.items():
        wav = os.path.join(args.swap_dir, f"{base}.wav")
        fspk, rspk = meta["face_spk"], meta["ref_spk"]
        if not (os.path.exists(wav) and fspk in centroids and rspk in centroids):
            skipped += 1; continue
        e = embed(wav)
        sr = cos(e, centroids[rspk]); sf = cos(e, centroids[fspk])
        sim_to_ref.append(sr); sim_to_face.append(sf)
        if sr > sf: ref_wins += 1
        else:       face_wins += 1

    n = ref_wins + face_wins
    print(f"\n[transfer] {n} swap utterances scored ({skipped} skipped)")
    print(f"  output closer to reference speaker  : {ref_wins} ({ref_wins/max(n,1):.3f})  <- transfer success")
    print(f"  output closer to face speaker : {face_wins} ({face_wins/max(n,1):.3f})  <- face-identity leak")
    print(f"  mean cosine to ref  speaker   : {np.mean(sim_to_ref):.3f}")
    print(f"  mean cosine to face speaker   : {np.mean(sim_to_face):.3f}")
    print(f"\n[interpret] transfer-success rate {ref_wins/max(n,1):.1%}. "
          f">0.5 means ref controls speaker (factorization works); "
          f"~0.5 means ambiguous; <0.5 means face identity leak dominates.")


if __name__ == "__main__":
    main()

