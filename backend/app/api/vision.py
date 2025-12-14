from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi import status as http_status

from app.schemas.vision import VisionPrediction, VisionResponse
from app.services.vision import classify_skin_image, vit_ready

router = APIRouter(prefix="/vision", tags=["vision"])


@router.post(
    "/skin-diagnosis",
    response_model=VisionResponse,
    status_code=http_status.HTTP_200_OK,
    summary="Classify a dermatology image with ViT",
)
async def diagnose_skin_lesion(
    file: UploadFile = File(..., description="Dermatology image to classify."),
    top_k: int = 3,
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
        disclaimer=(
            "Research demo only. This ViT classifier is not clinically validated and "
            "must not be used for diagnosis or treatment decisions."
        ),
    )
