"""FastAPI application entry point for EduPredict AI."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.recommendation import router as recommendation_router
from backend.schemas import HealthResponse
from backend.services.gemini_service import GeminiService
from backend.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load saved ML pipelines and Keras models once at application startup."""
    app.state.model_service.load_all()
    yield


app = FastAPI(
    title="EduPredict AI API",
    description=(
        "REST API backend for predicting student final grades (G3) using Scikit-learn "
        "and TensorFlow/Keras models, paired with Google Gemini personalized study plans."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow local frontend (Streamlit) and local dev origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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
    """Return service health, loaded model counts, and Gemini configuration status."""
    model_service: ModelService = app.state.model_service
    gemini_service: GeminiService = app.state.gemini_service
    model_service.ensure_loaded()

    return HealthResponse(
        status="healthy",
        ml_models_loaded=len(model_service.ml_pipelines),
        keras_models_loaded=len(model_service.keras_models),
        gemini_configured=gemini_service.is_configured(),
    )

