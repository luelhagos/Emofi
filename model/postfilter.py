"""
Intensity postfilter: a conditional residual refiner on the decoder mel.

A regression decoder predicts the conditional-mean mel, which over-smooths the
spectral texture that separates high-intensity from calm speech. The postfilter
learns a residual refined = mu + Delta with Delta = G(mu, style, level),
conditioned on the style vector and a discrete intensity level. It is trained on
a frozen backbone with an adversarial (multi-scale LSGAN) term, a per-level
global-variance term, a content anchor to mu, and the SER feature-matching loss
(model/serfm.py).

At inference the postfilter is applied to medium/high-intensity utterances and
bypassed for low-intensity ones (applying texture to calm speech over-energizes
it). The level embedding also supports a continuous control: a soft mixture over
the three learned level vectors (level_weights) interpolates intensity.

Shapes: mels are (B, n_mel, T) here (transpose the model's (B, T, n_mel)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Dilated residual conv block (leaky-relu, 'same' padding)."""

    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation)

    def forward(self, x):
        return x + F.leaky_relu(self.conv(x), 0.2)


class PostfilterGenerator(nn.Module):
    """Residual refiner: refined = mu + Delta, Delta = G(mu, style, level).

    out_proj is zero-initialised so G starts as the identity (refined == mu), a
    stable warm start that preserves intelligibility. The level enters as a
    learned embedding added to the conditioning before the nonlinear stack, so
    each level can select a distinct refinement rather than a scalar gain."""

    def __init__(self, n_mel=80, style_dim=128, hidden=256, n_layers=6,
                 dilation_cycle=3, level_cond=False):
        super().__init__()
        self.in_proj = nn.Conv1d(n_mel, hidden, 1)
        self.style_proj = nn.Linear(style_dim, hidden)
        self.level_emb = nn.Embedding(3, hidden) if level_cond else None
        self.blocks = nn.ModuleList([
            ConvBlock(hidden, 2 ** (i % dilation_cycle)) for i in range(n_layers)
        ])
        self.out_proj = nn.Conv1d(hidden, n_mel, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, mu, style, level=None, level_weights=None):
        # mu: (B, n_mel, T); style: (B, 1, style_dim); level: (B,) in {0,1,2}.
        # level_weights: (B, 3) soft mixture over the level embeddings -- a
        # continuous intensity control that overrides the discrete level.
        s = self.style_proj(style).transpose(1, 2)         # (B, hidden, 1)
        if self.level_emb is not None and level_weights is not None:
            s = s + (level_weights @ self.level_emb.weight).unsqueeze(-1)
        elif self.level_emb is not None and level is not None:
            s = s + self.level_emb(level).unsqueeze(-1)
        x = self.in_proj(mu) + s                            # broadcast over T
        for block in self.blocks:
            x = block(x)
        return mu + self.out_proj(x)                        # (B, n_mel, T)


class _ScaleDisc(nn.Module):
    """One strided-conv discriminator at a single temporal scale. Returns patch
    logits and the intermediate feature maps (for feature matching)."""

    def __init__(self, n_mel, hidden):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(n_mel, hidden, 5, 1, 2),
            nn.Conv1d(hidden, hidden, 5, 2, 2),
            nn.Conv1d(hidden, hidden, 5, 2, 2),
            nn.Conv1d(hidden, hidden, 5, 1, 2),
        ])
        self.out = nn.Conv1d(hidden, 1, 3, 1, 1)

    def forward(self, mel):
        x, feats = mel, []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.2)
            feats.append(x)
        return self.out(x), feats


class MelDiscriminator(nn.Module):
    """Multi-scale mel discriminator: real vs refined, at temporal downsamplings
    x1 / x2 / x4. Coarse scales judge envelope and dynamics, fine scales the
    texture. Unconditional; per-level targeting comes from the GV loss."""

    def __init__(self, n_mel=80, hidden=256, n_scales=3):
        super().__init__()
        self.discs = nn.ModuleList([_ScaleDisc(n_mel, hidden) for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(2, 2)

    def forward(self, mel):
        out, x = [], mel
        for d in self.discs:
            out.append(d(x))            # (logits, feats)
            x = self.pool(x)
        return out                       # list of (logits, feats)
