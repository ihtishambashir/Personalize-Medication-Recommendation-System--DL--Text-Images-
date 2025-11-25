from typing import List, Optional
from pydantic import BaseModel, Field


class MedicationSuggestion(BaseModel):
    drug_code: str = Field(..., description="Internal or ATC/RxNorm-like code")
    name: str = Field(..., description="Human-readable medication name")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    warnings: List[str] = Field(default_factory=list, description="Per-medication textual warnings")


class RecommendationRequest(BaseModel):
    patient_id: Optional[str] = Field(
        None, description="Optional anonymized patient identifier for logging or debugging."
    )
    diagnoses: List[str] = Field(default_factory=list, description="ICD-like diagnosis codes or free text.")
    procedures: List[str] = Field(default_factory=list, description="Optional procedure codes.")
    current_medications: List[str] = Field(
        default_factory=list,
        description="List of drug codes or names the patient is already taking.",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional free-text note (e.g., chief complaint, history).",
    )


class RecommendationResponse(BaseModel):
    suggestions: List[MedicationSuggestion]
    ddi_warnings: List[str] = Field(default_factory=list, description="DDI warnings for the full combination.")
    disclaimer: str = Field(
        ...,
        description="Strong disclaimer reminding that this is not for clinical use.",
    )
