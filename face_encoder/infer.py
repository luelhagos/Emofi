"""
Inference wrapper for the released face-to-emotion encoder checkpoint.

Loads face_encoder.pt (keys: model_config, model_state, epoch, val_loss) and
returns L2-normalized emotion embeddings for face images.

Usage:
    from face_encoder.infer import FaceEmotionInference

    inf = FaceEmotionInference("face_encoder/face_encoder.pt", device="cuda")
    emb = inf.get_embedding("face.png")                        # (768,)
    emb, intensity, emotion = inf.get_embedding_with_intensity("face.png")
"""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from face_encoder.model import FaceEmotionEncoder
except ImportError:  # run from inside face_encoder/
    from model import FaceEmotionEncoder

# label indices used by the released checkpoint's classification heads
EMOTION_MAP = {
    "angry": 0,
    "disgusted": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6,
    "contempt": 7,
}
INTENSITY_MAP = {"level1": 0, "level2": 1, "level3": 2}

INTENSITY_LABELS = {v: k for k, v in INTENSITY_MAP.items()}
EMOTION_LABELS = {v: k for k, v in EMOTION_MAP.items()}


class FaceEmotionInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        cfg = ckpt["model_config"]

        self.model = FaceEmotionEncoder(
            embedding_dim=cfg["embedding_dim"],
            num_emotions=cfg["num_emotions"],
            num_intensities=cfg["num_intensities"],
            pretrained=False,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"Loaded FaceEmotionEncoder from epoch {ckpt['epoch']} "
              f"(val_loss={ckpt['val_loss']:.4f})")

    @torch.no_grad()
    def get_embedding(self, face_path: str) -> torch.Tensor:
        """Return the L2-normalized emotion embedding (embedding_dim,)."""
        face = Image.open(face_path).convert("RGB")
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)
        emb, _, _ = self.model(face_tensor)
        return F.normalize(emb, dim=-1).squeeze(0)

    @torch.no_grad()
    def get_embedding_with_intensity(self, face_path: str) -> tuple:
        """Return (emotion_emb, predicted_intensity_label, predicted_emotion_label)."""
        face = Image.open(face_path).convert("RGB")
        face_tensor = self.transform(face).unsqueeze(0).to(self.device)
        emb, intensity_logits, emotion_logits = self.model(face_tensor)

        intensity_idx = intensity_logits.argmax(dim=-1).item()
        emotion_idx = emotion_logits.argmax(dim=-1).item()

        return (
            F.normalize(emb, dim=-1).squeeze(0),
            INTENSITY_LABELS[intensity_idx],
            EMOTION_LABELS[emotion_idx],
        )

    @torch.no_grad()
    def get_embedding_batch(self, face_paths: list) -> torch.Tensor:
        """Batch inference. Returns an (N, embedding_dim) tensor."""
        faces = []
        for p in face_paths:
            face = Image.open(p).convert("RGB")
            faces.append(self.transform(face))
        batch = torch.stack(faces).to(self.device)
        emb, _, _ = self.model(batch)
        return F.normalize(emb, dim=-1)
