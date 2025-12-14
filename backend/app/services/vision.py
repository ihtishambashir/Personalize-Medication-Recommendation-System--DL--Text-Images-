from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.models.vision_transformer import IMAGENET_MEAN, IMAGENET_STD, load_vit_checkpoint

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "trained_models" / "vision_transformer"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals initialised at import time for reuse across requests.
_vit_model = None
_vit_labels: List[str] = []
_transform = None

try:
    artifacts = load_vit_checkpoint(_MODEL_DIR, device=_DEVICE)
    _vit_model = artifacts.model
    _vit_labels = artifacts.idx_to_class

    image_size = int(artifacts.config.get("image_size", 224))
    mean = tuple(artifacts.config.get("mean", IMAGENET_MEAN))
    std = tuple(artifacts.config.get("std", IMAGENET_STD))
    _transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
except FileNotFoundError:
    # The model is optional; the API layer will return a friendly message if missing.
    _vit_model = None
    _transform = None
    _vit_labels = []
    logger.warning("Vision model artifacts not found in %s. Train the model before calling the endpoint.", _MODEL_DIR)
except Exception as exc:  # pragma: no cover - defensive fallback
    _vit_model = None
    _transform = None
    _vit_labels = []
    logger.exception("Failed to load vision model: %s", exc)


def vit_ready() -> bool:
    return _vit_model is not None and _transform is not None and bool(_vit_labels)


def _load_image(data: bytes) -> Image.Image:
    """Read raw bytes into a RGB PIL image."""
    with BytesIO(data) as buf:
        img = Image.open(buf)
        return img.convert("RGB")


def classify_skin_image(image_bytes: bytes, top_k: int = 3) -> List[Tuple[str, float]]:
    """Run ViT inference on an uploaded image and return top-k predictions."""
    if not vit_ready():
        return []

    try:
        image = _load_image(image_bytes)
    except (UnidentifiedImageError, OSError):
        return []

    tensor = _transform(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _vit_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(max(1, top_k), probs.numel())
    top_probs, top_indices = torch.topk(probs, k)

    results: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = _vit_labels[idx] if 0 <= idx < len(_vit_labels) else f"class_{idx}"
        results.append((label, float(prob)))
    return results
