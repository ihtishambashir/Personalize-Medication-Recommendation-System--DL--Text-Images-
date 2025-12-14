from typing import List, Optional

from pydantic import BaseModel, Field


class MedicationSuggestion(BaseModel):
    """Single medication suggestion with an associated score."""

    drug_code: str = Field(..., description="Internal or ATC/RxNorm-like code.")
    name: str = Field(..., description="Human-readable medication name.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0, 1].")
    warnings: List[str] = Field(
        default_factory=list,
        description="Per-medication textual warnings (e.g. simple rule-based flags).",
    )


class RecommendationRequest(BaseModel):
    """Input payload for the recommendation endpoint.

    This is intentionally compact but already reflects the multimodal nature of
    the system: structured codes, free text, and dermatology images.
    """

    patient_id: Optional[str] = Field(
        None,
        description="Optional anonymised patient identifier used only for logging or debugging.",
    )
    diagnoses: List[str] = Field(
        default_factory=list,
        description="ICD-like diagnosis codes or short diagnosis phrases.",
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="List of medications the patient is already taking.",
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Optional list of high-level symptom keywords.",
    )
    image_paths: List[str] = Field(
        default_factory=list,
        description=(
            "Local paths (relative to the backend) for dermatology images associated with the visit. "
            "In a real deployment this would typically be replaced by file uploads or an image service."
        ),
    )
    max_suggestions: int = Field(
        5,
        ge=1,
        le=20,
        description="How many medications should be returned in the suggestion list.",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional free-text clinical note (e.g. chief complaint, history).",
    )


class RecommendationResponse(BaseModel):
    suggestions: List[MedicationSuggestion]
    ddi_warnings: List[str] = Field(
        default_factory=list,
        description="DDI warnings for the full suggested combination.",
    )
    disclaimer: str = Field(
        ...,
        description="Strong disclaimer reminding that this is a research prototype and not for clinical use.",
    )
