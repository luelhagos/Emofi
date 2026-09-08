"""
Wav2Vec2-based speech emotion + intensity classifier (the SER judge).

Fine-tuned on MEAD audio; used at evaluation only to score synthesized wavs for
emotion (6-way) and intensity (3-way) accuracy. Architecture: a Wav2Vec2 encoder
with attentive pooling, an emotion head, and an emotion-conditioned ordinal
intensity head.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model



class OrdinalLoss(nn.Module):
    """
    Treats intensity as ordered thresholds:
        level1 ==> [0, 0]
        level2 ==> [1, 0]
        level3 ==> [1, 1]

    level2_weight applies extra loss to level2 samples on both thresholds,
    since level2 is confused in both directions (==> level1 and ==> level3).
    """

    NUM_THRESHOLDS = 2

    def __init__(self, level2_weight: float = 1.0):
        super().__init__()
        self.level2_weight = level2_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = labels.size(0)
        targets = torch.zeros(B, 2, device=labels.device)
        targets[labels >= 1, 0] = 1.0
        targets[labels >= 2, 1] = 1.0

        loss = self.bce(logits, targets)   # (B, 2)

        if self.level2_weight != 1.0:
            w = torch.ones(B, device=labels.device)
            w[labels == 1] = self.level2_weight
            loss = loss * w.unsqueeze(1)

        return loss.mean()

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().sum(dim=1)



class AttentivePooling(nn.Module):
    """
    Learns a scalar attention weight per time frame instead of averaging
    all frames equally. This lets the model focus on the most expressive
    moments in the utterance rather than diluting them with silence/neutral frames.

    Input  : (B, T, D)  hidden states from wav2vec2
    Output : (B, D)     weighted sum
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # single linear ==> scalar score per frame
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, hidden: torch.Tensor,
                bool_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hidden    : (B, T, D)
            bool_mask : (B, T) True for real frames, False for padding
        Returns:
            (B, D)
        """
        scores = self.attn(hidden).squeeze(-1)   # (B, T)

        if bool_mask is not None:
            # set padding frames to -inf so softmax ignores them
            scores = scores.masked_fill(~bool_mask, float("-inf"))

        weights = torch.softmax(scores, dim=1)    # (B, T)
        pooled  = (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return pooled



class EmotionClassifier(nn.Module):
    """
    Args:
        pretrained_name       : HuggingFace model id
        num_emotions          : number of emotion classes (default 6)
        num_levels            : number of intensity levels (default 3)
        label_mode            : "two_head" or "joint"
        num_joint             : joint class count (label_mode="joint" only)
        freeze_cnn            : freeze CNN feature extractor
        freeze_encoder_layers : freeze the first N transformer layers
        dropout               : dropout before heads
        emotion_context_dim   : projection size for emotion ==> intensity conditioning
        use_ordinal           : use ordinal intensity head (two_head only)
        level2_weight         : extra loss weight on level2 samples
        use_attentive_pool    : use attentive pooling instead of mean pooling
    """

    def __init__(
        self,
        pretrained_name: str = "facebook/wav2vec2-base",
        num_emotions: int = 6,
        num_levels: int = 3,
        label_mode: str = "two_head",
        num_joint: int = 16,
        freeze_cnn: bool = True,
        freeze_encoder_layers: int = 4,
        dropout: float = 0.2,
        emotion_context_dim: int = 128,
        use_ordinal: bool = True,
        level2_weight: float = 2.0,
        use_attentive_pool: bool = True,
    ):
        super().__init__()

        self.label_mode          = label_mode
        self.use_ordinal         = use_ordinal and (label_mode == "two_head")
        self.emotion_context_dim = emotion_context_dim
        self.num_emotions        = num_emotions
        self.use_attentive_pool  = use_attentive_pool

        self.encoder = Wav2Vec2Model.from_pretrained(pretrained_name)

        # freeze CNN
        if freeze_cnn:
            for p in self.encoder.feature_extractor.parameters():
                p.requires_grad = False
            for p in self.encoder.feature_projection.parameters():
                p.requires_grad = True

        # freeze bottom transformer layers
        if freeze_encoder_layers > 0:
            layers   = self.encoder.encoder.layers
            n_freeze = min(freeze_encoder_layers, len(layers))
            for layer in layers[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            print(f"[model] frozen: CNN + first {n_freeze}/{len(layers)} transformer layers")
        else:
            print("[model] frozen: CNN only")

        hidden = self.encoder.config.hidden_size   # 768 for wav2vec2-base

        # pooling
        if use_attentive_pool:
            self.pooling = AttentivePooling(hidden)
            print("[model] pooling: attentive")
        else:
            self.pooling = None
            print("[model] pooling: mean")

        self.dropout = nn.Dropout(dropout)

        if label_mode == "joint":
            self.joint_head = nn.Linear(hidden, num_joint)
        else:
            # emotion head
            self.emotion_head = nn.Linear(hidden, num_emotions)

            # intensity head (emotion-conditioned, deeper)
            # emotion logits ==> projection ==> concat with pooled ==> two-layer MLP
            self.intensity_proj = nn.Sequential(
                nn.Linear(num_emotions, emotion_context_dim),
                nn.ReLU(),
            )

            intensity_in  = hidden + emotion_context_dim
            intensity_out = OrdinalLoss.NUM_THRESHOLDS if self.use_ordinal else num_levels

            # deeper head with LayerNorm for better gradient flow
            self.intensity_head = nn.Sequential(
                nn.Linear(intensity_in, intensity_in // 2),
                nn.LayerNorm(intensity_in // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(intensity_in // 2, intensity_out),
            )

            self.ordinal_loss = OrdinalLoss(level2_weight=level2_weight)

        self._print_param_counts()

    # helpers

    def _print_param_counts(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] params -- total: {total/1e6:.1f}M  "
              f"trainable: {trainable/1e6:.1f}M  "
              f"frozen: {(total-trainable)/1e6:.1f}M")

    def _get_frame_mask(self, hidden, attention_mask):
        """Build boolean frame mask from raw attention mask."""
        if attention_mask is None:
            return None
        lengths   = self.encoder._get_feat_extract_output_lengths(
            attention_mask.sum(dim=1))
        bool_mask = torch.zeros(
            hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        for i, length in enumerate(lengths):
            bool_mask[i, :length] = True
        return bool_mask

    def _pool(self, hidden, bool_mask):
        """Pool (B, T, D) ==> (B, D) using attentive or mean pooling."""
        if self.use_attentive_pool:
            return self.pooling(hidden, bool_mask)
        # mean pool
        if bool_mask is not None:
            hidden  = hidden * bool_mask.unsqueeze(-1).float()
            pooled  = hidden.sum(dim=1) / bool_mask.sum(dim=1, keepdim=True).float()
        else:
            pooled = hidden.mean(dim=1)
        return pooled

    # forward

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Returns (label_mode="two_head"):
            emotion_logits   : (B, num_emotions)
            intensity_logits : (B, 2) if use_ordinal else (B, num_levels)

        Returns (label_mode="joint"):
            joint_logits : (B, num_joint)
        """
        outputs   = self.encoder(input_values=waveform, attention_mask=attention_mask)
        hidden    = outputs.last_hidden_state          # (B, T', 768)
        bool_mask = self._get_frame_mask(hidden, attention_mask)
        pooled    = self._pool(hidden, bool_mask)      # (B, 768)
        pooled    = self.dropout(pooled)

        if self.label_mode == "joint":
            return self.joint_head(pooled)

        # emotion head
        e_logits = self.emotion_head(pooled)           # (B, num_emotions)

        # intensity head -- conditioned on emotion, detached so
        # intensity loss does not affect emotion head gradients
        e_context = self.intensity_proj(e_logits.detach())    # (B, 128)
        i_input   = torch.cat([pooled, e_context], dim=-1)    # (B, 768+128)
        i_logits  = self.intensity_head(i_input)               # (B, 2) or (B, 3)

        return e_logits, i_logits

    # convenience

    def predict(self, waveform: torch.Tensor):
        self.eval()
        with torch.no_grad():
            out = self.forward(waveform)
        if self.label_mode == "joint":
            return out.argmax(dim=-1)
        e_logits, i_logits = out
        e_preds = e_logits.argmax(dim=-1)
        i_preds = (OrdinalLoss.predict(i_logits) if self.use_ordinal
                   else i_logits.argmax(dim=-1))
        return e_preds, i_preds



if __name__ == "__main__":
    model = EmotionClassifier(
        freeze_encoder_layers=4,
        use_ordinal=True,
        use_attentive_pool=True,
        emotion_context_dim=128,
    )
    dummy = torch.randn(2, 16000 * 4)
    e_out, i_out = model(dummy)
    print("emotion logits  :", e_out.shape)    # (2, 6)
    print("intensity logits:", i_out.shape)    # (2, 2)
    e_pred, i_pred = model.predict(dummy)
    print("emotion pred    :", e_pred)
    print("intensity pred  :", i_pred)