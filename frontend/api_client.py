"""HTTP client connecting the Streamlit frontend to the FastAPI backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load BACKEND_URL if configured in .env; never read or expose API keys in frontend
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


@dataclass
class ApiResult:
    """Standardized response wrapper for FastAPI HTTP calls."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


def _format_http_error(resp: requests.Response) -> str:
    """Extract a beginner-friendly error message from FastAPI HTTP error responses."""
    try:
        payload = resp.json()
    except ValueError:
        return f"Backend returned HTTP {resp.status_code}: {resp.text[:200]}"

    detail = payload.get("detail")
    if isinstance(detail, list):
        messages = []
        for item in detail:
            loc = " -> ".join(str(part) for part in item.get("loc", []))
            msg = item.get("msg", "Invalid input")
            messages.append(f"{loc}: {msg}" if loc else msg)
        return "Validation error: " + "; ".join(messages)
    if detail:
        return f"Backend error (HTTP {resp.status_code}): {detail}"
    return f"Backend returned HTTP {resp.status_code}."


def _request(
    method: str,
    url: str,
    json_payload: dict[str, Any] | None = None,
    timeout: int = 15,
) -> ApiResult:
    """Execute an HTTP request and handle connection, timeout, and HTTP errors cleanly."""
    try:
        resp = requests.request(method=method, url=url, json=json_payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        base = url.rsplit("/", 1)[0]
        return ApiResult(
            ok=False,
            error=(
                f"Cannot connect to FastAPI backend at `{base}`. "
                "Start it with: `uvicorn backend.main:app --reload`"
            ),
        )
    except requests.exceptions.Timeout:
        return ApiResult(
            ok=False,
            error="The backend request timed out. Please check if the server is busy and retry.",
        )
    except requests.exceptions.RequestException as exc:
        return ApiResult(ok=False, error=f"Unexpected network error: {exc}")

    if resp.ok:
        try:
            return ApiResult(ok=True, data=resp.json(), status_code=resp.status_code)
        except ValueError:
            return ApiResult(
                ok=False,
                error="Backend returned an invalid JSON response.",
                status_code=resp.status_code,
            )

    return ApiResult(
        ok=False,
        error=_format_http_error(resp),
        status_code=resp.status_code,
    )


def check_health(base_url: str = DEFAULT_BACKEND_URL) -> ApiResult:
    """Call GET /health to check FastAPI availability and loaded model counts."""
    return _request("GET", f"{base_url.rstrip('/')}/health", timeout=3)


def fetch_models(base_url: str = DEFAULT_BACKEND_URL) -> ApiResult:
    """Call GET /models to retrieve registered ML and Keras model metadata."""
    return _request("GET", f"{base_url.rstrip('/')}/models", timeout=5)


def predict_grade(payload: dict[str, Any], base_url: str = DEFAULT_BACKEND_URL) -> ApiResult:
    """Call POST /predict to compute student G3 predictions."""
    return _request("POST", f"{base_url.rstrip('/')}/predict", json_payload=payload, timeout=15)


def get_recommendations(payload: dict[str, Any], base_url: str = DEFAULT_BACKEND_URL) -> ApiResult:
    """Call POST /recommendations to compute prediction and Gemini study plan."""
    return _request(
        "POST",
        f"{base_url.rstrip('/')}/recommendations",
        json_payload=payload,
        timeout=45,
    )

