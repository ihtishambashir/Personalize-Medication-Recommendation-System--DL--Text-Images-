from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def create_vit_model(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.1,
) -> nn.Module:
    """Build a torchvision ViT backbone with a custom classification head."""
    try:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    except AttributeError:
        weights = None
    model = models.vit_b_16(weights=weights)

    # Replace the classification head so it matches our dataset.
    in_features = model.heads.head.in_features
    head_layers = [nn.LayerNorm(in_features)]
    if dropout is not None and dropout > 0:
        head_layers.append(nn.Dropout(dropout))
    head_layers.append(nn.Linear(in_features, num_classes))
    model.heads = nn.Sequential(*head_layers)

    return model


def save_vit_checkpoint(
    output_dir: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    config: Dict[str, object],
) -> None:
    """Persist model weights plus metadata needed for inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state": model.state_dict(),
        "config": config,
    }
    torch.save(ckpt, output_dir / "vision_transformer.pt")

    # Label mapping is needed to turn logits into class names.
    (output_dir / "classes.json").write_text(
        json.dumps(class_to_idx, indent=2, sort_keys=True)
    )


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

    class_to_idx = json.loads(classes_path.read_text())
    idx_to_class = [""] * len(class_to_idx)
    for label, idx in class_to_idx.items():
        idx_to_class[int(idx)] = label

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {})
    num_classes = len(idx_to_class)
    dropout = float(config.get("dropout", 0.1))

    model = create_vit_model(
        num_classes=num_classes,
        pretrained=False,  # weights come from the checkpoint
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return VisionArtifacts(model=model, idx_to_class=idx_to_class, config=config)
