"""Prediction and model catalog routes for EduPredict AI."""

from fastapi import APIRouter, HTTPException, Request, status

from backend.schemas import (
    ModelsCatalogResponse,
    PredictionRequest,
    PredictionResponse,
)
from backend.services.model_service import KerasDisabledError, KerasUnavailableError

router = APIRouter(tags=["Predictions"])


@router.get(
    "/models",
    response_model=ModelsCatalogResponse,
    summary="List available ML and Keras models",
)
def list_models(request: Request) -> ModelsCatalogResponse:
    """Return metadata and unseen test metrics for all 4 prediction tasks."""
    model_service = request.app.state.model_service
    try:
        tasks = model_service.get_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Saved model files are not available on the server.",
        ) from exc

    return ModelsCatalogResponse(total_tasks=len(tasks), tasks=tasks)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict student final grade (G3)",
)
def predict_grade(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Predict final grade (`G3`, 0-20) using the best classical ML pipeline by default,
    or optionally compare against the saved Keras Neural Network (`model_type='keras'` or `'both'`).
    """
    model_service = request.app.state.model_service
    try:
        return model_service.predict(
            subject=payload.subject,
            scenario=payload.scenario,
            student=payload.student,
            model_type=payload.model_type,
        )
    except KerasDisabledError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except KerasUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Requested prediction task not found: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model inference failed unexpectedly.",
        ) from exc

