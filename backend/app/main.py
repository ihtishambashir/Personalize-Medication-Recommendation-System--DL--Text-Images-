from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as recommend_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Research-only demo API for a Personalized Medication Recommendation System (PMRS).\n\n"
        "⚠️ This service is NOT a medical device and MUST NOT be used for real clinical decisions."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend_router, prefix=settings.api_prefix)


@app.get("/", tags=["health"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "message": "PMRS demo backend is running. Not for clinical use.",
    }
