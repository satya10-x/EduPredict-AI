"""Automated Streamlit frontend tests using AppTest and FastAPI TestClient."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from backend.main import app as fastapi_app

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _bridge_requests_to_test_client(monkeypatch):
    """Route frontend requests.request calls through an in-process FastAPI TestClient."""
    client = TestClient(fastapi_app)
    client.__enter__()

    def fake_request(method: str, url: str, json=None, timeout=15):
        path = "/" + url.split("://", 1)[1].split("/", 1)[1] if "://" in url else url
        resp = client.request(method=method, url=path, json=json)
        return SimpleNamespace(
            ok=resp.is_success,
            status_code=resp.status_code,
            text=resp.text,
            json=resp.json,
        )

    monkeypatch.setattr("frontend.api_client.requests.request", fake_request)
    return client


def test_overview_page_when_backend_offline(monkeypatch):
    """Overview page displays Backend Offline status cleanly without crashing."""
    import requests as real_requests

    def offline_request(*args, **kwargs):
        raise real_requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr("frontend.api_client.requests.request", offline_request)

    at = AppTest.from_file(APP_PATH, default_timeout=20)
    at.run()
    assert not at.exception
    assert any("Backend Offline" in err.value for err in at.error)


def test_full_frontend_flow_with_fastapi_backend(monkeypatch):
    """Verify Overview, Student Prediction (all 4 scenarios), Recommendations, and Model Comparison."""
    client = _bridge_requests_to_test_client(monkeypatch)
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=25)
        at.run()
        assert not at.exception
        assert any("Backend Online" in s.value for s in at.sidebar.success)

        # 1. Navigate to Student Prediction
        at.sidebar.radio[0].set_value("Student Prediction").run()
        assert not at.exception

        # Test all 4 (subject, scenario) combinations through the live FastAPI /predict endpoint
        for subj in ["Mathematics", "Portuguese"]:
            for scen_label in [
                "early — Before term grades (excludes G1 & G2)",
                "late — Mid-year prediction (requires G1 & G2)",
            ]:
                at.selectbox[1].set_value(subj).run()
                at.selectbox[2].set_value(scen_label).run()
                at.selectbox[3].set_value("both — Compare ML vs Keras Side-by-Side").run()
                at.button[1].click().run()  # Form submit button ('Predict Final Grade (G3)')
                assert not at.exception
                assert at.session_state["latest_prediction"] is not None
                pred = at.session_state["latest_prediction"]
                assert pred["subject"] == subj
                assert "ml" in pred["comparison"] and "keras" in pred["comparison"]

        # 2. Navigate to AI Study Recommendations and call /recommendations
        at.sidebar.radio[0].set_value("AI Study Recommendations").run()
        assert not at.exception
        at.button[0].click().run()  # 'Generate AI Study Plan'
        assert not at.exception
        assert at.session_state["latest_recommendation"] is not None
        assert at.session_state["latest_recommendation"]["prediction_result"] is not None

        # 3. Navigate to Model Comparison and verify /models + live Actual vs Predicted button
        at.sidebar.radio[0].set_value("Model Comparison").run()
        assert not at.exception
        at.button[0].click().run()  # 'Run Live Actual vs Predicted Evaluation'
        assert not at.exception
        assert len(at.session_state["live_actual_vs_pred"]) >= 1
    finally:
        client.__exit__(None, None, None)


def test_backend_url_resolution_env_and_secrets(monkeypatch):
    """Verify BACKEND_URL resolves from env, st.secrets, or localhost fallback."""
    from frontend.api_client import LOCAL_FALLBACK_URL, get_default_backend_url

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr("frontend.api_client.st.secrets", {"BACKEND_URL": "https://edupredict-ai-vfnx.onrender.com/"})
    assert get_default_backend_url() == "https://edupredict-ai-vfnx.onrender.com"

    monkeypatch.setenv("BACKEND_URL", "http://127.0.0.1:9000/")
    assert get_default_backend_url() == "http://127.0.0.1:9000"

    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setattr("frontend.api_client.st.secrets", {})
    assert get_default_backend_url() == LOCAL_FALLBACK_URL


def test_cold_start_timeout_and_gateway_errors(monkeypatch):
    """Render cold-start timeouts and 502/503 HTML errors produce understandable messages."""
    import requests as real_requests
    from frontend.api_client import check_health, predict_grade

    def timeout_request(*args, **kwargs):
        raise real_requests.exceptions.Timeout("Read timed out")

    monkeypatch.setattr("frontend.api_client.requests.request", timeout_request)
    res_timeout = predict_grade({"dummy": 1}, base_url="https://edupredict-ai-vfnx.onrender.com")
    assert not res_timeout.ok
    assert "Render Free Tier" in res_timeout.error

    def gateway_503(*args, **kwargs):
        return SimpleNamespace(
            ok=False,
            status_code=503,
            text="<html>Service Unavailable</html>",
            json=lambda: (_ for _ in ()).throw(ValueError("not json")),
        )

    monkeypatch.setattr("frontend.api_client.requests.request", gateway_503)
    res_503 = check_health(base_url="https://edupredict-ai-vfnx.onrender.com")
    assert not res_503.ok
    assert "HTTP 503" in res_503.error
    assert "waking up from sleep" in res_503.error


def test_frontend_with_ml_only_backend(monkeypatch):
    """Frontend handles ENABLE_KERAS_MODELS=false backend cleanly."""
    monkeypatch.setenv("ENABLE_KERAS_MODELS", "false")
    client = _bridge_requests_to_test_client(monkeypatch)
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=25)
        at.run()
        assert not at.exception
        assert any("Backend Online" in s.value for s in at.sidebar.success)
        assert at.session_state["keras_enabled"] is False

        at.sidebar.radio[0].set_value("Student Prediction").run()
        assert not at.exception
        at.button[1].click().run()
        assert not at.exception
        assert at.session_state["latest_prediction"]["prediction"]["model_family"] == "ml"
    finally:
        client.__exit__(None, None, None)

