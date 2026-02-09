from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch
from torch import nn
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VisionArtifacts:
    """Container returned by the loader for convenient access."""

    model: nn.Module
    idx_to_class: List[str]
    config: Dict[str, object]


def _resolve_vit_weights(pretrained: bool):
    """Return torchvision ViT weights when available for this version."""
    if not pretrained:
        return None
    try:
        return models.ViT_B_16_Weights.IMAGENET1K_V1
    except AttributeError:
        # Older torchvision versions may not expose enum-style weights.
        return None


def create_vit_model(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.1,
) -> nn.Module:
    """Create a ViT-B/16 classifier with a dataset-specific head."""
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}.")

    weights = _resolve_vit_weights(pretrained)
    model = models.vit_b_16(weights=weights)

    head_layer = getattr(model.heads, "head", None)
    if head_layer is None or not hasattr(head_layer, "in_features"):
        raise AttributeError("Unexpected torchvision ViT head structure; expected heads.head.in_features.")

    # Replace the classification head so output logits match our class count.
    in_features = int(head_layer.in_features)
    head_layers = [nn.LayerNorm(in_features)]
    if dropout is not None and float(dropout) > 0:
        head_layers.append(nn.Dropout(float(dropout)))
    head_layers.append(nn.Linear(in_features, num_classes))
    model.heads = nn.Sequential(*head_layers)

    return model


def save_vit_checkpoint(
    output_dir: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    config: Dict[str, object],
) -> None:
    """Persist model weights and metadata needed for inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "config": dict(config or {}),
    }
    torch.save(ckpt, output_dir / "vision_transformer.pt")

    # Label mapping is needed to turn logits into class names.
    class_to_idx_payload = {str(label): int(idx) for label, idx in class_to_idx.items()}
    (output_dir / "classes.json").write_text(
        json.dumps(class_to_idx_payload, indent=2, sort_keys=True)
    )


def _normalise_class_to_idx(raw_mapping: Mapping[str, object]) -> Dict[str, int]:
    class_to_idx: Dict[str, int] = {}
    for raw_label, raw_idx in raw_mapping.items():
        label = str(raw_label).strip()
        if not label:
            continue
        idx = int(raw_idx)
        if idx < 0:
            raise ValueError(f"Invalid class index for label '{label}': {idx}")
        class_to_idx[label] = idx
    if not class_to_idx:
        raise ValueError("classes.json did not contain any valid class mapping.")
    return class_to_idx


def _build_idx_to_class(class_to_idx: Mapping[str, int]) -> List[str]:
    max_idx = max(int(idx) for idx in class_to_idx.values())
    idx_to_class = [""] * (max_idx + 1)

    for label, idx in class_to_idx.items():
        idx = int(idx)
        if idx < len(idx_to_class) and not idx_to_class[idx]:
            idx_to_class[idx] = str(label)

    # If an index is missing, we keep a deterministic fallback name.
    for idx, label in enumerate(idx_to_class):
        if not label:
            idx_to_class[idx] = f"class_{idx}"
    return idx_to_class


def load_vit_checkpoint(
    model_dir: Path,
    device: torch.device | str = "cpu",
) -> VisionArtifacts:
    """Load a trained ViT checkpoint from disk for inference."""
    model_dir = Path(model_dir)

    ckpt_path = model_dir / "vision_transformer.pt"
    classes_path = model_dir / "classes.json"
    if not (ckpt_path.exists() and classes_path.exists()):
        raise FileNotFoundError(
            f"Missing vision checkpoint files in {model_dir}. "
            "Expected vision_transformer.pt and classes.json."
        )

    raw_mapping = json.loads(classes_path.read_text())
    if not isinstance(raw_mapping, dict):
        raise ValueError("classes.json must be a JSON object mapping class name -> index.")
    class_to_idx = _normalise_class_to_idx(raw_mapping)
    idx_to_class = _build_idx_to_class(class_to_idx)

    ckpt = torch.load(ckpt_path, map_location=device)
    config_raw = ckpt.get("config", {})
    config = dict(config_raw) if isinstance(config_raw, dict) else {}
    num_classes = len(idx_to_class)
    dropout = float(config.get("dropout", 0.1))

    model = create_vit_model(
        num_classes=num_classes,
        pretrained=False,  # weights come from the checkpoint
        dropout=dropout,
    )
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise KeyError("Checkpoint is missing a valid 'model_state' entry.")
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return VisionArtifacts(model=model, idx_to_class=idx_to_class, config=config)


def get_vit_patch_size(model: nn.Module) -> Tuple[int, int]:
    """Best-effort extraction of the ViT patch size (height, width)."""
    conv_proj = getattr(model, "conv_proj", None)
    if conv_proj is None:
        return (16, 16)

    kernel_size = getattr(conv_proj, "kernel_size", (16, 16))
    if isinstance(kernel_size, tuple):
        if len(kernel_size) == 2:
            return int(kernel_size[0]), int(kernel_size[1])
        if len(kernel_size) == 1:
            return int(kernel_size[0]), int(kernel_size[0])

    value = int(kernel_size)
    return value, value


def get_vit_encoder_depth(model: nn.Module) -> int:
    """Return the number of transformer encoder blocks if available."""
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return 0
    return int(len(layers))


def get_vit_num_heads(model: nn.Module) -> int:
    """Return attention head count of the first encoder block if available."""
    encoder = getattr(model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if not layers:
        return 0

    first_layer = layers[0]
    attn = getattr(first_layer, "self_attention", None)
    if attn is None:
        attn = getattr(first_layer, "self_attn", None)
    if attn is None:
        return 0
    return int(getattr(attn, "num_heads", 0))
