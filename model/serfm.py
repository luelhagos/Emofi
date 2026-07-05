"""
SER feature-matching loss for the intensity postfilter.

An emotion-intensity recognizer reads high intensity largely from spectral
texture (harmonicity, spectral flux, energy above 4 kHz) that a regression
decoder over-smooths. This module trains the postfilter to match that texture:
the refined mel is vocoded and its intermediate-layer features under a frozen
self-supervised speech model are matched (L1) to those of the same utterance's
reference audio. Intermediate layers encode acoustic texture rather than class
labels, so this avoids the reward-hacking failure mode of maximizing recognizer
logits.

The feature extractor should be a different model from any recognizer used for
evaluation, so that gains must transfer across models. The chain is
refined mel -> vocoder -> waveform -> SSL features; vocoder and SSL are frozen
but gradients flow through them into the postfilter generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SERFeatureMatcher(nn.Module):
    """Frozen SSL feature extractor + L1 feature matching at an intermediate layer.

    The caller supplies waveforms (from vocoder(mel)); this module handles
    resampling to the SSL rate, per-utterance normalization, and extraction."""

    def __init__(self, model_name, layer=4, in_sr=22050, ssl_sr=16000):
        super().__init__()
        from transformers import AutoModel
        self.ssl = AutoModel.from_pretrained(model_name)
        self.ssl.eval()
        for p in self.ssl.parameters():
            p.requires_grad = False
        self.layer = layer
        self.in_sr, self.ssl_sr = in_sr, ssl_sr
        self._resampler = None    # lazy torchaudio resampler, built on first use

    def _to_ssl_input(self, wav):
        """(B, samples) at in_sr -> (B, samples') at ssl_sr, zero-mean unit-var."""
        if self.in_sr != self.ssl_sr:
            if self._resampler is None:
                import torchaudio
                self._resampler = torchaudio.transforms.Resample(
                    self.in_sr, self.ssl_sr).to(wav.device)
            wav = self._resampler(wav)
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / (wav.std(dim=-1, keepdim=True) + 1e-7)
        return wav

    def features(self, wav):
        x = self._to_ssl_input(wav)
        out = self.ssl(x, output_hidden_states=True)
        return out.hidden_states[self.layer]      # (B, T', H)

    def fm_loss(self, refined_wav, gt_wav):
        """L1 feature matching at the chosen layer. Gradient flows from
        refined_wav only; the ground-truth path is the fixed target."""
        f_ref = self.features(refined_wav)
        with torch.no_grad():
            f_gt = self.features(gt_wav)
        T = min(f_ref.shape[1], f_gt.shape[1])
        return F.l1_loss(f_ref[:, :T], f_gt[:, :T])
