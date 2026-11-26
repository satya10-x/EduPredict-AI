"""AI study recommendation routes combining ML predictions with Google Gemini."""

from fastapi import APIRouter, HTTPException, Request, status

from backend.schemas import RecommendationRequest, RecommendationResponse
from backend.services.model_service import KerasDisabledError, KerasUnavailableError

router = APIRouter(tags=["Recommendations"])


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict final grade and generate AI study plan",
)
def get_recommendations(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
    """
    Run ML/Keras grade prediction and pass the student profile + predicted score
    to Google Gemini for a personalized study plan.

    If Gemini API is unavailable or unconfigured, the endpoint still returns the
    ML/Keras prediction alongside a clear `gemini_status` message.
    """
    model_service = request.app.state.model_service
    gemini_service = request.app.state.gemini_service

    try:
        prediction_result = model_service.predict(
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
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute student grade prediction.",
        ) from exc

    gemini_status, plan, message = gemini_service.generate_study_plan(
        student=payload.student,
        prediction=prediction_result,
    )

    return RecommendationResponse(
        subject=payload.subject,
        scenario=payload.scenario,
        prediction_result=prediction_result,
        gemini_status=gemini_status,
        gemini_model=gemini_service.model_name if gemini_status == "success" else None,
        recommendations=plan,
        message=message,
    )

