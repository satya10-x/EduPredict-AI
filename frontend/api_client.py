"""HTTP client connecting the Streamlit frontend to the FastAPI backend."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Load BACKEND_URL if configured in local .env; never read or expose API keys in frontend
load_dotenv(PROJECT_ROOT / ".env", override=False)

LOCAL_FALLBACK_URL = "http://127.0.0.1:8000"


def get_default_backend_url() -> str:
    """
    Resolve backend URL from environment variable or Streamlit secrets,
    preserving http://127.0.0.1:8000 as the local development default.
    """
    env_url = os.getenv("BACKEND_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")

    try:
        secret_url = str(st.secrets.get("BACKEND_URL", "")).strip()
        if secret_url:
            return secret_url.rstrip("/")
    except Exception:
        # No .streamlit/secrets.toml present locally or outside Streamlit runtime
        pass

    return LOCAL_FALLBACK_URL


DEFAULT_BACKEND_URL = get_default_backend_url()


@dataclass
class ApiResult:
    """Standardized response wrapper for FastAPI HTTP calls."""

    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    status_code: int | None = None


def _format_http_error(resp: requests.Response) -> str:
    """Extract a beginner-friendly error message from FastAPI or gateway error responses."""
    try:
        payload = resp.json()
    except ValueError:
        if resp.status_code in (502, 503, 504):
            return (
                f"Backend temporarily unavailable (HTTP {resp.status_code}). "
                "If deployed on Render Free Tier, the server may be waking up from sleep — "
                "please wait 30–60 seconds and retry."
            )
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
    timeout: int = 30,
) -> ApiResult:
    """Execute an HTTP request and handle connection, cold-start timeout, and HTTP errors."""
    try:
        resp = requests.request(method=method, url=url, json=json_payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        base = url.rsplit("/", 1)[0]
        return ApiResult(
            ok=False,
            error=(
                f"Cannot connect to FastAPI backend at `{base}`. "
                "Verify the backend URL is correct and the FastAPI server is running."
            ),
        )
    except requests.exceptions.Timeout:
        return ApiResult(
            ok=False,
            error=(
                "The backend request timed out. If deployed on Render Free Tier, the server "
                "may be waking up from sleep (30–60s) — please wait a moment and retry."
            ),
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


def check_health(base_url: str | None = None) -> ApiResult:
    """Call GET /health to check FastAPI availability and loaded model counts."""
    target = (base_url or get_default_backend_url()).rstrip("/")
    return _request("GET", f"{target}/health", timeout=20)


def fetch_models(base_url: str | None = None) -> ApiResult:
    """Call GET /models to retrieve registered ML and Keras model metadata."""
    target = (base_url or get_default_backend_url()).rstrip("/")
    return _request("GET", f"{target}/models", timeout=30)


def predict_grade(payload: dict[str, Any], base_url: str | None = None) -> ApiResult:
    """Call POST /predict to compute student G3 predictions."""
    target = (base_url or get_default_backend_url()).rstrip("/")
    return _request("POST", f"{target}/predict", json_payload=payload, timeout=60)


def get_recommendations(payload: dict[str, Any], base_url: str | None = None) -> ApiResult:
    """Call POST /recommendations to compute prediction and Gemini study plan."""
    target = (base_url or get_default_backend_url()).rstrip("/")
    return _request(
        "POST",
        f"{target}/recommendations",
        json_payload=payload,
        timeout=60,
    )
