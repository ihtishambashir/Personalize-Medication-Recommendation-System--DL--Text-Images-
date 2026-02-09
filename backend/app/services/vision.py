from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.models.vision_transformer import IMAGENET_MEAN, IMAGENET_STD, load_vit_checkpoint

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "trained_models" / "vision_transformer"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Globals initialised at import time for reuse across requests.
_vit_model: torch.nn.Module | None = None
_vit_labels: List[str] = []
_transform: transforms.Compose | None = None
_vit_config: Dict[str, object] = {}
_image_size = 224
_mean = tuple(IMAGENET_MEAN)
_std = tuple(IMAGENET_STD)

try:
    artifacts = load_vit_checkpoint(_MODEL_DIR, device=_DEVICE)
    _vit_model = artifacts.model
    _vit_labels = artifacts.idx_to_class
    _vit_config = dict(artifacts.config or {})

    _image_size = int(_vit_config.get("image_size", 224))
    _mean = tuple(_vit_config.get("mean", IMAGENET_MEAN))
    _std = tuple(_vit_config.get("std", IMAGENET_STD))
    _transform = transforms.Compose(
        [
            transforms.Resize(_image_size + 32),
            transforms.CenterCrop(_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std),
        ]
    )
except FileNotFoundError:
    # The model is optional; the API layer will return a friendly message if missing.
    _vit_model = None
    _transform = None
    _vit_labels = []
    _vit_config = {}
    _image_size = 224
    _mean = tuple(IMAGENET_MEAN)
    _std = tuple(IMAGENET_STD)
    logger.warning("Vision model artifacts not found in %s. Train the model before calling the endpoint.", _MODEL_DIR)
except Exception as exc:  # pragma: no cover - defensive fallback
    _vit_model = None
    _transform = None
    _vit_labels = []
    _vit_config = {}
    _image_size = 224
    _mean = tuple(IMAGENET_MEAN)
    _std = tuple(IMAGENET_STD)
    logger.exception("Failed to load vision model: %s", exc)


def vit_ready() -> bool:
    return _vit_model is not None and _transform is not None and bool(_vit_labels)


def get_vit_runtime() -> Dict[str, Any] | None:
    """Expose loaded ViT runtime artifacts for advanced analysis modules."""
    if not vit_ready():
        return None
    return {
        "model": _vit_model,
        "labels": list(_vit_labels),
        "transform": _transform,
        "device": _DEVICE,
        "config": dict(_vit_config),
        "image_size": int(_image_size),
        "mean": tuple(_mean),
        "std": tuple(_std),
    }


def decode_image_bytes(data: bytes) -> Image.Image:
    """Read raw bytes into a RGB PIL image."""
    with BytesIO(data) as buf:
        img = Image.open(buf)
        return img.convert("RGB")


def preprocess_skin_image(image_bytes: bytes) -> torch.Tensor | None:
    """Decode and transform raw image bytes into a model-ready batch tensor."""
    if not vit_ready():
        return None

    try:
        image = decode_image_bytes(image_bytes)
    except (UnidentifiedImageError, OSError):
        return None

    assert _transform is not None  # guarded by vit_ready()
    return _transform(image).unsqueeze(0).to(_DEVICE)


def predict_skin_logits(image_tensor: torch.Tensor) -> torch.Tensor | None:
    """Run a forward pass on a preprocessed image tensor."""
    if not vit_ready():
        return None
    assert _vit_model is not None
    with torch.no_grad():
        return _vit_model(image_tensor)


def classify_skin_image(image_bytes: bytes, top_k: int = 3) -> List[Tuple[str, float]]:
    """Run ViT inference on an uploaded image and return top-k predictions."""
    tensor = preprocess_skin_image(image_bytes)
    if tensor is None:
        return []

    logits = predict_skin_logits(tensor)
    if logits is None:
        return []
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(max(1, top_k), probs.numel())
    top_probs, top_indices = torch.topk(probs, k)

    results: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = _vit_labels[idx] if 0 <= idx < len(_vit_labels) else f"class_{idx}"
        results.append((label, float(prob)))
    return results
