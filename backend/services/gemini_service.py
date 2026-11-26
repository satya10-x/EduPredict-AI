"""Google Gemini API service for generating personalized student study plans."""

import json
import logging
import os
from pathlib import Path
from typing import Tuple

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from backend.schemas import PredictionResponse, StudentFeatures, StudyPlan

PLACEHOLDER_KEY = "your_gemini_api_key_here"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
logger = logging.getLogger("edupredict.gemini")


def _sanitize_error(message: str, api_key: str | None) -> str:
    """Ensure API key substrings are never logged or returned."""
    if api_key and api_key in message:
        return message.replace(api_key, "[REDACTED]")
    return message


class GeminiService:
    """Safely connects to Google Gemini using GEMINI_API_KEY from .env."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.env_path = self.project_root / ".env"
        self._env_mtime: float | None = None
        if self.env_path.is_file():
            # At startup, load .env so stale shell values do not override updated project config
            load_dotenv(self.env_path, override=True)
            self._env_mtime = self.env_path.stat().st_mtime
        self.model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL

    def _refresh_env_if_changed(self) -> None:
        """Reload .env automatically if the file on disk was modified since last check."""
        if not self.env_path.is_file():
            return
        current_mtime = self.env_path.stat().st_mtime
        if self._env_mtime != current_mtime:
            load_dotenv(self.env_path, override=True)
            self._env_mtime = current_mtime

    def _get_api_key(self) -> str | None:
        """Read GEMINI_API_KEY from environment; return None if missing or placeholder."""
        self._refresh_env_if_changed()
        raw_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        if not raw_key or raw_key == PLACEHOLDER_KEY:
            return None
        return raw_key

    def is_configured(self) -> bool:
        """Return True if a non-placeholder GEMINI_API_KEY is available."""
        return self._get_api_key() is not None

    def _use_vertex_ai(self) -> bool:
        """Return True only when Vertex AI routing is explicitly requested."""
        return os.getenv("GEMINI_USE_VERTEXAI", "").strip().lower() == "true"

    def _create_client(self, api_key: str) -> Tuple[genai.Client, str]:
        """
        Instantiate google-genai Client for Google AI Studio by default, or
        Vertex AI only when GEMINI_USE_VERTEXAI=true is explicitly set.
        """
        if self._use_vertex_ai():
            return genai.Client(vertexai=True, api_key=api_key), "Vertex"
        return genai.Client(api_key=api_key), "AI Studio"

    def _build_prompt(
        self,
        student: StudentFeatures,
        prediction: PredictionResponse,
    ) -> str:
        """Build a concise, beginner-friendly prompt containing student context and ML prediction."""
        pred = prediction.prediction
        profile = {
            "subject": prediction.subject,
            "prediction_scenario": prediction.scenario,
            "predicted_final_grade_out_of_20": pred.predicted_grade,
            "performance_band": pred.performance_band,
            "passed_threshold_10": pred.passed,
            "studytime_level_1_to_4": student.studytime,
            "past_failures": student.failures,
            "absences": student.absences,
            "school_support": student.schoolsup,
            "family_support": student.famsup,
            "extra_paid_classes": student.paid,
            "wants_higher_education": student.higher,
            "internet_at_home": student.internet,
            "free_time_1_to_5": student.freetime,
            "going_out_1_to_5": student.goout,
            "G1": student.G1,
            "G2": student.G2,
        }

        return (
            "You are an encouraging academic advisor for a high school student.\n"
            "Based on the student's attributes and machine learning prediction below, "
            "generate a clear, practical, personalized study plan.\n\n"
            f"Student Data & Prediction:\n{json.dumps(profile, indent=2)}\n\n"
            "Keep recommendations specific, realistic, and supportive."
        )

    def _classify_api_error(self, exc: genai_errors.APIError, api_key: str, key_type: str) -> str:
        """Log and translate google-genai APIError codes into safe user messages."""
        code = getattr(exc, "code", None)
        status = str(getattr(exc, "status", "") or "")
        safe_detail = _sanitize_error(str(getattr(exc, "message", None) or exc), api_key)

        logger.warning(
            "Gemini APIError (key_type=%s, code=%s, status=%s): %s",
            key_type,
            code,
            status,
            safe_detail,
        )

        if code == 401 or status == "UNAUTHENTICATED":
            return (
                f"Gemini authentication failed (401 UNAUTHENTICATED, key_type={key_type}): "
                "Verify GEMINI_API_KEY in .env is valid. ML prediction remains available."
            )
        if code == 403 or status == "PERMISSION_DENIED":
            return (
                f"Gemini API permission denied (403 PERMISSION_DENIED, key_type={key_type}): "
                "Check API key permissions or Google Cloud API enablement. ML prediction remains available."
            )
        if code == 404 or status == "NOT_FOUND":
            return (
                f"Configured Gemini model '{self.model_name}' was not found (404 NOT_FOUND). "
                f"Try GEMINI_MODEL={DEFAULT_GEMINI_MODEL}. ML prediction remains available."
            )
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return (
                "Gemini API quota or rate limit exceeded (429 RESOURCE_EXHAUSTED). "
                "Please wait and retry later. ML prediction remains available."
            )
        if code == 503 or status == "UNAVAILABLE":
            return (
                f"Gemini model '{self.model_name}' is temporarily overloaded (503 UNAVAILABLE). "
                "Please retry shortly. ML prediction remains available."
            )
        return f"Gemini API error (HTTP {code} {status}). ML prediction remains available."

    def generate_study_plan(
        self,
        student: StudentFeatures,
        prediction: PredictionResponse,
    ) -> Tuple[str, StudyPlan | None, str | None]:
        """
        Call Google Gemini to generate structured study recommendations.

        Returns:
            (status, recommendations, message)
            - status: 'success', 'unavailable', or 'error'
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.info("Gemini API key is not configured; returning unavailable status.")
            return (
                "unavailable",
                None,
                "Gemini API key is not configured. Copy .env.example to .env and set GEMINI_API_KEY.",
            )

        model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        self.model_name = model_name
        prompt = self._build_prompt(student, prediction)
        key_type = "Vertex" if self._use_vertex_ai() else "AI Studio"

        try:
            client, key_type = self._create_client(api_key)
            logger.info("Calling Gemini API | key_type=%s | model=%s", key_type, model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    response_mime_type="application/json",
                    response_schema=StudyPlan,
                ),
            )
            raw_text = response.text or ""
            plan = StudyPlan.model_validate_json(raw_text)
            return ("success", plan, None)
        except genai_errors.APIError as exc:
            return ("error", None, self._classify_api_error(exc, api_key, key_type))
        except (httpx.HTTPError, ConnectionError, TimeoutError) as exc:
            logger.warning(
                "Network error reaching Gemini API (key_type=%s, exc=%s): %s",
                key_type,
                type(exc).__name__,
                _sanitize_error(str(exc), api_key),
            )
            return (
                "error",
                None,
                "Network error while contacting Gemini API. Check internet connectivity. "
                "ML prediction remains available.",
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse structured StudyPlan from Gemini response: %s", type(exc).__name__)
            return (
                "error",
                None,
                "Gemini response format was invalid. ML prediction remains available.",
            )
        except Exception as exc:
            logger.error(
                "Unexpected error calling Gemini API (key_type=%s, exc=%s): %s",
                key_type,
                type(exc).__name__,
                _sanitize_error(str(exc), api_key),
            )
            return (
                "error",
                None,
                f"Failed to generate recommendations from Gemini API ({type(exc).__name__}). "
                "ML prediction remains available.",
            )
