from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from PIL import Image, ImageDraw

from app.models.vision_transformer import (
    get_vit_encoder_depth,
    get_vit_num_heads,
    get_vit_patch_size,
)
from app.services.vision import get_vit_runtime, preprocess_skin_image, vit_ready


@dataclass(frozen=True)
class SalientPatch:
    rank: int
    row: int
    col: int
    importance: float


@dataclass(frozen=True)
class VisionPatternAnalysis:
    predicted_label: str
    confidence: float
    top_k: List[Tuple[str, float]]
    image_size: int
    patch_size: int
    patch_grid_rows: int
    patch_grid_cols: int
    num_patches: int
    transformer_layers: int
    attention_heads: int
    recognition_steps: List[str]
    salient_patches: List[SalientPatch]
    pattern_summary: str


@dataclass(frozen=True)
class VisionOverlayResult:
    predicted_label: str
    confidence: float
    image_png: bytes


def _extract_top_k(
    probs: torch.Tensor,
    labels: List[str],
    top_k: int,
) -> List[Tuple[str, float]]:
    k = min(max(1, int(top_k)), probs.numel())
    top_probs, top_indices = torch.topk(probs, k)

    results: List[Tuple[str, float]] = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = labels[idx] if 0 <= idx < len(labels) else f"class_{idx}"
        results.append((label, float(prob)))
    return results


def _infer_with_saliency(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one forward/backward pass and return (probs, saliency_map)."""
    sample = image_tensor.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)

    logits = model(sample)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    top_index = int(torch.argmax(logits, dim=1).item())
    logits[0, top_index].backward()

    if sample.grad is None:
        h = int(sample.shape[-2])
        w = int(sample.shape[-1])
        saliency_map = torch.zeros((h, w), device=sample.device, dtype=sample.dtype)
        return probs.detach(), saliency_map

    # Absolute gradient averaged across channels is a simple, stable saliency score.
    saliency = sample.grad.detach().abs().mean(dim=1, keepdim=True)  # (1, 1, H, W)
    saliency = F.avg_pool2d(saliency, kernel_size=7, stride=1, padding=3)
    saliency = saliency.squeeze(0).squeeze(0)  # (H, W)

    saliency = saliency - saliency.min()
    saliency = saliency / saliency.max().clamp(min=1e-8)
    return probs.detach(), saliency.detach()


def _compute_salient_patches_from_map(
    saliency_map: torch.Tensor,
    patch_h: int,
    patch_w: int,
    max_hotspots: int = 3,
) -> List[SalientPatch]:
    patch_h = max(1, int(patch_h))
    patch_w = max(1, int(patch_w))

    pooled = F.avg_pool2d(
        saliency_map.unsqueeze(0).unsqueeze(0),
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w),
    )
    pooled_map = pooled.squeeze(0).squeeze(0)  # (rows, cols)
    if pooled_map.numel() == 0:
        return []

    k = min(max(1, int(max_hotspots)), pooled_map.numel())
    values, indices = torch.topk(pooled_map.flatten(), k)
    cols = int(pooled_map.size(1))

    hotspots: List[SalientPatch] = []
    for rank, (value, flat_idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        row = int(flat_idx // cols)
        col = int(flat_idx % cols)
        hotspots.append(
            SalientPatch(
                rank=rank,
                row=row,
                col=col,
                importance=float(value),
            )
        )
    return hotspots


def _denormalize_image(
    image_tensor: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> np.ndarray:
    """Convert normalized tensor (1,3,H,W) into RGB float image in [0,1]."""
    img = image_tensor[0].detach().cpu()
    mean_t = torch.tensor(mean, dtype=img.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=img.dtype).view(3, 1, 1)
    img = (img * std_t + mean_t).clamp(0.0, 1.0)
    img_np = img.permute(1, 2, 0).numpy()
    return img_np


def _build_overlay_image(
    base_image: np.ndarray,
    saliency_map: torch.Tensor,
    salient_patches: List[SalientPatch],
    patch_h: int,
    patch_w: int,
    predicted_label: str,
    confidence: float,
) -> bytes:
    """Create a realistic color overlay image that highlights diagnosis regions."""
    saliency = saliency_map.detach().cpu().numpy()
    saliency = np.clip(saliency, 0.0, 1.0)

    # Colorize with a high-contrast heatmap and blend using per-pixel alpha.
    heatmap = cm.get_cmap("turbo")(saliency)[..., :3]
    alpha_map = (saliency ** 0.7) * 0.7
    alpha_map = np.expand_dims(alpha_map, axis=-1)

    overlay = base_image * (1.0 - alpha_map) + heatmap * alpha_map
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8), mode="RGB")

    draw = ImageDraw.Draw(overlay_img)
    box_colors = ["#00FF66", "#FFFF66", "#FFB347"]
    img_w, img_h = overlay_img.size

    # Mark top salient patches with bright borders.
    for patch in salient_patches:
        color = box_colors[(patch.rank - 1) % len(box_colors)]
        x0 = int(patch.col * patch_w)
        y0 = int(patch.row * patch_h)
        x1 = min(img_w - 1, x0 + int(patch_w) - 1)
        y1 = min(img_h - 1, y0 + int(patch_h) - 1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0 + 2, y0 + 2), f"#{patch.rank}", fill=color)

    # Add prediction text strip.
    caption = f"Pred: {predicted_label}  Conf: {confidence:.1%}"
    try:
        text_bbox = draw.textbbox((0, 0), caption)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except AttributeError:
        text_w = 180
        text_h = 16

    pad = 6
    strip = [0, 0, min(img_w - 1, text_w + pad * 2), min(img_h - 1, text_h + pad * 2)]
    draw.rectangle(strip, fill=(0, 0, 0))
    draw.text((pad, pad), caption, fill=(255, 255, 255))

    with BytesIO() as buf:
        overlay_img.save(buf, format="PNG")
        return buf.getvalue()


def _build_recognition_steps(
    image_size: int,
    patch_size: int,
    patch_grid_rows: int,
    patch_grid_cols: int,
    transformer_layers: int,
    attention_heads: int,
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
) -> List[str]:
    return [
        "Decode uploaded JPG/PNG and convert to RGB.",
        f"Resize and center-crop to {image_size}x{image_size} pixels.",
        f"Normalize pixels with mean={mean} and std={std}.",
        (
            f"Split image into {patch_grid_rows}x{patch_grid_cols} patches "
            f"(patch size {patch_size}x{patch_size})."
        ),
        "Project each patch to an embedding token and add a [CLS] token.",
        (
            f"Run tokens through {transformer_layers} Transformer encoder layers "
            f"with {attention_heads} attention heads."
        ),
        "Use the classification head to produce class logits, then softmax for probabilities.",
    ]


def _build_pattern_summary(
    predicted_label: str,
    confidence: float,
    patch_grid_rows: int,
    patch_grid_cols: int,
    salient_patches: List[SalientPatch],
) -> str:
    if not salient_patches:
        return (
            f"Predicted '{predicted_label}' with confidence {confidence:.3f}. "
            "A saliency hotspot map could not be computed for this image."
        )

    strongest = salient_patches[0]
    return (
        f"Predicted '{predicted_label}' with confidence {confidence:.3f}. "
        f"Strongest evidence was near patch row {strongest.row}, column {strongest.col} "
        f"in a {patch_grid_rows}x{patch_grid_cols} patch grid."
    )


def analyze_skin_pattern(
    image_bytes: bytes,
    top_k: int = 3,
) -> VisionPatternAnalysis | None:
    """Run ViT inference and return an explanation of image pattern recognition."""
    if not vit_ready():
        return None

    runtime = get_vit_runtime()
    if runtime is None:
        return None

    model = runtime["model"]
    labels: List[str] = runtime["labels"]
    image_size = int(runtime["image_size"])
    mean = tuple(runtime["mean"])
    std = tuple(runtime["std"])

    image_tensor = preprocess_skin_image(image_bytes)
    if image_tensor is None:
        return None

    probs, saliency_map = _infer_with_saliency(model, image_tensor)

    top_predictions = _extract_top_k(probs, labels, top_k)
    if not top_predictions:
        return None

    predicted_label, confidence = top_predictions[0]

    patch_h, patch_w = get_vit_patch_size(model)
    patch_size = int(min(patch_h, patch_w))
    patch_grid_rows = max(1, image_size // max(1, patch_h))
    patch_grid_cols = max(1, image_size // max(1, patch_w))
    num_patches = patch_grid_rows * patch_grid_cols

    transformer_layers = get_vit_encoder_depth(model)
    attention_heads = get_vit_num_heads(model)
    salient_patches = _compute_salient_patches_from_map(
        saliency_map=saliency_map,
        patch_h=patch_h,
        patch_w=patch_w,
        max_hotspots=3,
    )
    recognition_steps = _build_recognition_steps(
        image_size=image_size,
        patch_size=patch_size,
        patch_grid_rows=patch_grid_rows,
        patch_grid_cols=patch_grid_cols,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        mean=mean,
        std=std,
    )
    pattern_summary = _build_pattern_summary(
        predicted_label=predicted_label,
        confidence=confidence,
        patch_grid_rows=patch_grid_rows,
        patch_grid_cols=patch_grid_cols,
        salient_patches=salient_patches,
    )

    return VisionPatternAnalysis(
        predicted_label=predicted_label,
        confidence=float(confidence),
        top_k=top_predictions,
        image_size=image_size,
        patch_size=patch_size,
        patch_grid_rows=patch_grid_rows,
        patch_grid_cols=patch_grid_cols,
        num_patches=num_patches,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads,
        recognition_steps=recognition_steps,
        salient_patches=salient_patches,
        pattern_summary=pattern_summary,
    )


def render_skin_pattern_overlay(
    image_bytes: bytes,
    top_k: int = 3,
) -> VisionOverlayResult | None:
    """Return a color diagnostic overlay PNG that highlights model focus regions."""
    if not vit_ready():
        return None

    runtime = get_vit_runtime()
    if runtime is None:
        return None

    model = runtime["model"]
    labels: List[str] = runtime["labels"]
    mean = tuple(runtime["mean"])
    std = tuple(runtime["std"])

    image_tensor = preprocess_skin_image(image_bytes)
    if image_tensor is None:
        return None

    probs, saliency_map = _infer_with_saliency(model, image_tensor)
    top_predictions = _extract_top_k(probs, labels, top_k)
    if not top_predictions:
        return None

    predicted_label, confidence = top_predictions[0]
    patch_h, patch_w = get_vit_patch_size(model)
    salient_patches = _compute_salient_patches_from_map(
        saliency_map=saliency_map,
        patch_h=patch_h,
        patch_w=patch_w,
        max_hotspots=3,
    )

    base_image = _denormalize_image(image_tensor, mean=mean, std=std)
    image_png = _build_overlay_image(
        base_image=base_image,
        saliency_map=saliency_map,
        salient_patches=salient_patches,
        patch_h=patch_h,
        patch_w=patch_w,
        predicted_label=predicted_label,
        confidence=float(confidence),
    )
    return VisionOverlayResult(
        predicted_label=predicted_label,
        confidence=float(confidence),
        image_png=image_png,
    )
