from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class VisionPrediction(BaseModel):
    label: str = Field(..., description="Predicted skin condition label.")
    score: float = Field(..., ge=0.0, le=1.0, description="Softmax confidence in [0, 1].")


class VisionResponse(BaseModel):
    predicted_label: str = Field(..., description="Top-1 predicted class.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the top prediction.")
    top_k: List[VisionPrediction] = Field(
        ...,
        description="Top-k predictions sorted by confidence.",
    )
    disclaimer: str = Field(
        ...,
        description="Reminder that this model is a research prototype and not a medical device.",
    )


class VisionSalientPatch(BaseModel):
    rank: int = Field(..., ge=1, description="Rank in the saliency hotspot list.")
    row: int = Field(..., ge=0, description="Patch row index (0-based).")
    col: int = Field(..., ge=0, description="Patch column index (0-based).")
    importance: float = Field(..., ge=0.0, le=1.0, description="Relative saliency score for this patch.")


class VisionPatternModelInfo(BaseModel):
    architecture: str = Field(..., description="Backbone architecture used for prediction.")
    image_size: int = Field(..., ge=1, description="Input image size after preprocessing.")
    patch_size: int = Field(..., ge=1, description="Patch side length used by ViT.")
    patch_grid_rows: int = Field(..., ge=1, description="Number of patch rows.")
    patch_grid_cols: int = Field(..., ge=1, description="Number of patch columns.")
    num_patches: int = Field(..., ge=1, description="Total number of patches fed into ViT.")
    transformer_layers: int = Field(..., ge=0, description="Number of encoder layers in the ViT.")
    attention_heads: int = Field(..., ge=0, description="Attention heads per encoder layer.")


class VisionPatternResponse(BaseModel):
    predicted_label: str = Field(..., description="Top-1 predicted class.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the top prediction.")
    top_k: List[VisionPrediction] = Field(..., description="Top-k predictions sorted by confidence.")
    model_info: VisionPatternModelInfo = Field(
        ...,
        description="Static ViT architecture details for this inference run.",
    )
    recognition_steps: List[str] = Field(
        ...,
        description="Step-by-step explanation of how the model processed the image.",
    )
    salient_patches: List[VisionSalientPatch] = Field(
        default_factory=list,
        description="Most influential patch locations for the top prediction.",
    )
    pattern_summary: str = Field(
        ...,
        description="Human-readable summary of the recognized pattern evidence.",
    )
    disclaimer: str = Field(
        ...,
        description="Reminder that this model is a research prototype and not a medical device.",
    )
