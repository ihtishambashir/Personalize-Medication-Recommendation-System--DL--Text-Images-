from fastapi import APIRouter

from app.schemas.medication import RecommendationRequest, RecommendationResponse
from app.services.recommendation import recommend_medications

router = APIRouter(prefix="/recommend", tags=["recommendation"])


@router.post("", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest) -> RecommendationResponse:
    """Return a set of demo medication suggestions with DDI safety warnings.

    In your thesis you can later:
    - replace the toy model with your multimodal architecture,
    - integrate image features,
    - plug in a real DDI knowledge graph, etc.
    """
    return recommend_medications(request)
