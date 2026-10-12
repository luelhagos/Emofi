"""
Face-to-emotion encoder architecture.

Maps a face image to an emotion embedding in the emotion2vec space, with
auxiliary heads for emotion and intensity classification. Only the class
definition is needed to load the trained checkpoint (face_encoder.pt).
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class FaceEmotionEncoder(nn.Module):
    """
    Face image -> emotion embedding (embedding_dim, default 768).

    Also produces:
      - intensity logits (3-way: level1/2/3)
      - emotion logits  (num_emotions-way)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_emotions: int = 8,
        num_intensities: int = 3,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Visual backbone: ResNet50 without the final FC (avgpool -> (B, 2048))
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        self.visual_encoder = nn.Sequential(*list(backbone.children())[:-1])
        visual_dim = 2048

        # MLP projector to the emotion embedding
        self.emotion_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(visual_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

        # Intensity head (branches off the emotion embedding)
        self.intensity_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_intensities),
        )

        # Emotion classification head (auxiliary)
        self.emotion_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions),
        )

    def forward(self, face_image: torch.Tensor):
        """
        Args:
            face_image: (B, 3, H, W)
        Returns:
            emotion_emb:      (B, embedding_dim)
            intensity_logits: (B, num_intensities)
            emotion_logits:   (B, num_emotions)
        """
        visual_feat = self.visual_encoder(face_image)       # (B, 2048, 1, 1)
        emotion_emb = self.emotion_projector(visual_feat)   # (B, embedding_dim)

        intensity_logits = self.intensity_head(emotion_emb)
        emotion_logits = self.emotion_head(emotion_emb)

        return emotion_emb, intensity_logits, emotion_logits

    def get_emotion_embedding(self, face_image: torch.Tensor) -> torch.Tensor:
        """Inference-only: return just the embedding."""
        emotion_emb, _, _ = self.forward(face_image)
        return emotion_emb
