from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from fastapi import status as http_status

from app.schemas.vision import (
    VisionPatternModelInfo,
    VisionPatternResponse,
    VisionPrediction,
    VisionResponse,
    VisionSalientPatch,
)
from app.services.vision import classify_skin_image, vit_ready
from app.services.vision_pattern import analyze_skin_pattern, render_skin_pattern_overlay

router = APIRouter(prefix="/vision", tags=["vision"])

_RESEARCH_ONLY_DISCLAIMER = (
    "Research demo only. This ViT classifier is not clinically validated and "
    "must not be used for diagnosis or treatment decisions."
)


@router.post(
    "/skin-diagnosis",
    response_model=VisionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Classify a dermatology image with ViT",
)
async def diagnose_skin_lesion(
    file: UploadFile = File(..., description="Dermatology image to classify."),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to return."),
) -> VisionResponse:
    """Return the top-k ViT predictions for an uploaded skin image."""
    if not vit_ready():
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model not available. Train it with train_vit_skin.py first.",
        )

    image_bytes = await file.read()
    predictions = classify_skin_image(image_bytes, top_k=top_k)
    if not predictions:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Could not read or classify the provided image. Please upload a valid JPG/PNG file.",
        )

    top_label, top_score = predictions[0]
    return VisionResponse(
        predicted_label=top_label,
        confidence=top_score,
        top_k=[VisionPrediction(label=label, score=score) for label, score in predictions],
        disclaimer=_RESEARCH_ONLY_DISCLAIMER,
    )


@router.post(
    "/skin-pattern-analysis",
    response_model=VisionPatternResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Upload image and inspect how ViT recognized the pattern",
)
async def explain_skin_pattern(
    file: UploadFile = File(..., description="Dermatology image to classify and explain."),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to return."),
) -> VisionPatternResponse:
    """Return top-k predictions plus a model-level pattern-recognition explanation."""
    if not vit_ready():
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model not available. Train it with train_vit_skin.py first.",
        )

    image_bytes = await file.read()
    analysis = analyze_skin_pattern(image_bytes, top_k=top_k)
    if analysis is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Could not read or classify the provided image. Please upload a valid JPG/PNG file.",
        )

    return VisionPatternResponse(
        predicted_label=analysis.predicted_label,
        confidence=analysis.confidence,
        top_k=[VisionPrediction(label=label, score=score) for label, score in analysis.top_k],
        model_info=VisionPatternModelInfo(
            architecture="torchvision.vit_b_16",
            image_size=analysis.image_size,
            patch_size=analysis.patch_size,
            patch_grid_rows=analysis.patch_grid_rows,
            patch_grid_cols=analysis.patch_grid_cols,
            num_patches=analysis.num_patches,
            transformer_layers=analysis.transformer_layers,
            attention_heads=analysis.attention_heads,
        ),
        recognition_steps=analysis.recognition_steps,
        salient_patches=[
            VisionSalientPatch(
                rank=patch.rank,
                row=patch.row,
                col=patch.col,
                importance=patch.importance,
            )
            for patch in analysis.salient_patches
        ],
        pattern_summary=analysis.pattern_summary,
        disclaimer=_RESEARCH_ONLY_DISCLAIMER,
    )


@router.post(
    "/skin-pattern-analysis/image",
    status_code=http_status.HTTP_200_OK,
    summary="Return colored diagnosis-area image from ViT pattern analysis",
)
async def explain_skin_pattern_image(
    file: UploadFile = File(..., description="Dermatology image to classify and visualize."),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to consider."),
) -> StreamingResponse:
    """Return PNG overlay highlighting image regions used by the model."""
    if not vit_ready():
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model not available. Train it with train_vit_skin.py first.",
        )

    image_bytes = await file.read()
    overlay = render_skin_pattern_overlay(image_bytes, top_k=top_k)
    if overlay is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Could not read or classify the provided image. Please upload a valid JPG/PNG file.",
        )

    headers = {
        "X-Predicted-Label": overlay.predicted_label,
        "X-Confidence": f"{overlay.confidence:.6f}",
    }
    return StreamingResponse(
        BytesIO(overlay.image_png),
        media_type="image/png",
        headers=headers,
    )
