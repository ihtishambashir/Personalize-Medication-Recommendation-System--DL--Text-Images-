from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as recommend_router
from app.core.config import get_settings

settings = get_settings()


def create_app() -> FastAPI:
    """Application factory.

    Keeping this in a small helper makes it easier to plug the API into tests
    or other ASGI servers later on.
    """
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description=(
            "Research-only backend API for a Personalized Medication Recommendation System (PMRS).\n\n"
            "⚠️ This service is not a medical device and must not be used for real clinical decisions."
        ),
    )

    # For a real deployment you should replace "*" with the concrete frontend URL(s).
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # All recommendation endpoints live under /api/recommend/...
    application.include_router(recommend_router, prefix=settings.api_prefix)
    return application


app = create_app()


@app.get("/", tags=["health"])
async def health_check() -> dict:
    """Simple health-check endpoint used by the frontend and tests."""
    return {
        "status": "ok",
        "message": "PMRS backend is running (research-only, not for clinical use).",
    }
