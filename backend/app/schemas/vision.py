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
