"""FastAPI application entry point for EduPredict AI."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.recommendation import router as recommendation_router
from backend.schemas import HealthResponse
from backend.services.gemini_service import GeminiService
from backend.services.model_service import ModelService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


def get_allowed_origins() -> list[str]:
    """Read comma-separated ALLOWED_ORIGINS from env, falling back to localhost defaults."""
    raw = os.getenv("ALLOWED_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_ALLOWED_ORIGINS)
    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved classical ML pipelines once at startup (Keras loads lazily if enabled)."""
    app.state.model_service.load_all()
    yield


app = FastAPI(
    title="EduPredict AI API",
    description=(
        "REST API backend for predicting student final grades (G3) using Scikit-learn "
        "and optional TensorFlow/Keras models, paired with Google Gemini personalized study plans."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configurable CORS origins; credentials remain disabled for safety
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Attach shared service instances on app.state
app.state.model_service = ModelService()
app.state.gemini_service = GeminiService()

# Mount modular route groups
app.include_router(prediction_router)
app.include_router(recommendation_router)


@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Health check")
def health_check() -> HealthResponse:
    """Return service health, loaded model counts, Keras mode, and Gemini status."""
    model_service: ModelService = app.state.model_service
    gemini_service: GeminiService = app.state.gemini_service
    model_service.ensure_loaded()

    return HealthResponse(
        status="healthy",
        ml_models_loaded=len(model_service.ml_pipelines),
        keras_models_loaded=len(model_service.keras_models),
        keras_enabled=model_service.is_keras_enabled(),
        gemini_configured=gemini_service.is_configured(),
    )
