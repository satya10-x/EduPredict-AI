"""Automated pytest suite for the EduPredict AI FastAPI backend."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app


def sample_student_payload(include_grades: bool = True) -> dict:
    """Return a valid UCI student payload for API tests."""
    data = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 3,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "no",
        "activities": "yes",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 2,
        "Dalc": 1,
        "Walc": 1,
        "health": 4,
        "absences": 4,
    }
    if include_grades:
        data["G1"] = 13
        data["G2"] = 14
    return data


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with FastAPI lifespan startup/shutdown executed."""
    with TestClient(app) as test_client:
        yield test_client


def test_saved_model_files_exist():
    """Verify all saved ML pipelines and Keras models exist in models/."""
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    expected_files = [
        "math_early_pipeline.joblib",
        "math_late_pipeline.joblib",
        "portuguese_early_pipeline.joblib",
        "portuguese_late_pipeline.joblib",
        "math_early_nn.keras",
        "math_late_nn.keras",
        "portuguese_early_nn.keras",
        "portuguese_late_nn.keras",
        "math_early_nn_preprocessor.joblib",
        "math_late_nn_preprocessor.joblib",
        "portuguese_early_nn_preprocessor.joblib",
        "portuguese_late_nn_preprocessor.joblib",
    ]
    for filename in expected_files:
        assert (models_dir / filename).is_file(), f"Missing model file: {filename}"


def test_health_endpoint_and_no_key_exposure(client, monkeypatch):
    """GET /health returns loaded counts and never exposes GEMINI_API_KEY."""
    secret_key = "secret-test-key-never-expose"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["ml_models_loaded"] == 4
    assert body["keras_models_loaded"] == 4
    assert body["gemini_configured"] is True
    assert secret_key not in resp.text


def test_swagger_docs_and_cors(client):
    """Swagger UI (/docs), OpenAPI schema, and CORS headers work properly."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    schema_resp = client.get("/openapi.json")
    assert schema_resp.status_code == 200
    paths = schema_resp.json()["paths"]
    for route in ["/health", "/models", "/predict", "/recommendations"]:
        assert route in paths

    cors_resp = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert cors_resp.status_code == 200
    assert cors_resp.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_models_catalog_endpoint(client):
    """GET /models lists all 4 subject-scenario tasks and metrics."""
    resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_tasks"] == 4
    pairs = {(t["subject"], t["scenario"]) for t in body["tasks"]}
    assert pairs == {
        ("Mathematics", "early"),
        ("Mathematics", "late"),
        ("Portuguese", "early"),
        ("Portuguese", "late"),
    }


@pytest.mark.parametrize(
    "subject,scenario,expected_default_model",
    [
        ("Mathematics", "early", "Random Forest Regressor"),
        ("Mathematics", "late", "Random Forest Regressor"),
        ("Portuguese", "early", "Random Forest Regressor"),
        ("Portuguese", "late", "Linear Regression"),
    ],
)
def test_predict_all_four_tasks_default_ml(client, subject, scenario, expected_default_model):
    """POST /predict works for all 4 tasks using the best Day 3 ML model by default."""
    include_grades = scenario == "late"
    payload = {
        "subject": subject,
        "scenario": scenario,
        "student": sample_student_payload(include_grades=include_grades),
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["subject"] == subject
    assert body["scenario"] == scenario
    assert body["requested_model_type"] == "ml"
    assert body["default_production_model"] == expected_default_model

    pred = body["prediction"]
    assert pred["model_family"] == "ml"
    assert pred["model_name"] == expected_default_model
    assert 0.0 <= pred["predicted_grade"] <= 20.0
    assert isinstance(pred["raw_prediction"], float)
    assert body["comparison"] is None


def test_predict_keras_and_both_comparison(client):
    """POST /predict supports model_type='keras' and model_type='both'."""
    base_student = sample_student_payload(include_grades=True)

    keras_resp = client.post(
        "/predict",
        json={"subject": "Mathematics", "scenario": "late", "model_type": "keras", "student": base_student},
    )
    assert keras_resp.status_code == 200
    keras_body = keras_resp.json()
    assert keras_body["prediction"]["model_family"] == "keras"
    assert keras_body["prediction"]["model_name"] == "Keras Sequential NN"

    both_resp = client.post(
        "/predict",
        json={"subject": "Portuguese", "scenario": "late", "model_type": "both", "student": base_student},
    )
    assert both_resp.status_code == 200
    both_body = both_resp.json()
    assert both_body["prediction"]["model_family"] == "ml"
    assert "ml" in both_body["comparison"]
    assert "keras" in both_body["comparison"]
    assert 0.0 <= both_body["comparison"]["keras"]["predicted_grade"] <= 20.0


def test_input_validation_errors(client):
    """Invalid inputs and missing G1/G2 for late scenario return HTTP 422."""
    # Late scenario without G1/G2
    missing_grades = client.post(
        "/predict",
        json={
            "subject": "Mathematics",
            "scenario": "late",
            "student": sample_student_payload(include_grades=False),
        },
    )
    assert missing_grades.status_code == 422

    # Out-of-range age and invalid school code
    bad_student = sample_student_payload(include_grades=False)
    bad_student["age"] = 99
    bad_student["school"] = "INVALID"
    bad_resp = client.post(
        "/predict",
        json={"subject": "Mathematics", "scenario": "early", "student": bad_student},
    )
    assert bad_resp.status_code == 422


def test_predict_works_without_gemini_key(client, monkeypatch):
    """ML prediction works independently even when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    resp = client.post(
        "/predict",
        json={
            "subject": "Mathematics",
            "scenario": "early",
            "student": sample_student_payload(include_grades=False),
        },
    )
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["prediction"]["predicted_grade"] <= 20.0


def test_recommendations_when_gemini_unavailable(client, monkeypatch):
    """POST /recommendations still returns ML prediction when Gemini key is missing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    resp = client.post(
        "/recommendations",
        json={
            "subject": "Portuguese",
            "scenario": "early",
            "student": sample_student_payload(include_grades=False),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_status"] == "unavailable"
    assert body["recommendations"] is None
    assert body["prediction_result"]["prediction"]["model_family"] == "ml"
    assert 0.0 <= body["prediction_result"]["prediction"]["predicted_grade"] <= 20.0


def test_recommendations_when_gemini_errors_never_leaks_key(client, monkeypatch):
    """Upstream Gemini failure returns graceful status without exposing API key."""
    secret_key = "fake-secret-gemini-key-12345"
    monkeypatch.setenv("GEMINI_API_KEY", secret_key)

    def broken_client(api_key: str):
        raise RuntimeError(f"Simulated network failure with key={api_key}")

    monkeypatch.setattr("backend.services.gemini_service.genai.Client", broken_client)

    resp = client.post(
        "/recommendations",
        json={
            "subject": "Mathematics",
            "scenario": "late",
            "student": sample_student_payload(include_grades=True),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_status"] == "error"
    assert body["recommendations"] is None
    assert body["prediction_result"]["prediction"]["predicted_grade"] >= 0.0
    assert secret_key not in resp.text


def test_recommendations_with_mocked_gemini_success(client, monkeypatch):
    """POST /recommendations returns structured StudyPlan when Gemini succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-valid-key-for-mock")

    mock_plan = {
        "summary": "Student shows strong readiness to pass Mathematics.",
        "strengths": ["Consistent weekly study time", "Zero past failures"],
        "risk_factors": ["4 recorded absences"],
        "weekly_study_plan": ["Review class notes twice weekly", "Practice 10 algebra problems daily"],
        "action_items": ["Attend upcoming review session"],
    }

    class FakeModels:
        def generate_content(self, model, contents, config):
            assert "Mathematics" in contents
            return SimpleNamespace(text=json.dumps(mock_plan))

    class FakeClient:
        def __init__(self, api_key: str):
            self.models = FakeModels()

    monkeypatch.setattr("backend.services.gemini_service.genai.Client", FakeClient)

    resp = client.post(
        "/recommendations",
        json={
            "subject": "Mathematics",
            "scenario": "late",
            "student": sample_student_payload(include_grades=True),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gemini_status"] == "success"
    assert body["recommendations"]["summary"] == mock_plan["summary"]
    assert len(body["recommendations"]["weekly_study_plan"]) == 2
    assert body["prediction_result"]["prediction"]["model_family"] == "ml"


def test_recommendations_api_error_classification_and_endpoint_routing(client, monkeypatch):
    """AQ. keys default to AI Studio unless GEMINI_USE_VERTEXAI=true, and errors never leak keys."""
    from google.genai import errors as genai_errors

    aq_secret = "AQ.FakeAIStudioKeyForUnitTest1234567890"
    monkeypatch.setenv("GEMINI_API_KEY", aq_secret)
    monkeypatch.delenv("GEMINI_USE_VERTEXAI", raising=False)

    captured_kwargs = {}

    class FakeModelsError:
        def generate_content(self, model, contents, config):
            raise genai_errors.APIError(
                403,
                {"error": {"code": 403, "message": f"Permission denied for {aq_secret}", "status": "PERMISSION_DENIED"}},
            )

    class FakeClientCapture:
        def __init__(self, **kwargs):
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
            self.models = FakeModelsError()

    monkeypatch.setattr("backend.services.gemini_service.genai.Client", FakeClientCapture)

    # 1. Without GEMINI_USE_VERTEXAI=true, AQ. key must route to AI Studio (no vertexai=True)
    resp = client.post(
        "/recommendations",
        json={
            "subject": "Mathematics",
            "scenario": "early",
            "student": sample_student_payload(include_grades=False),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "vertexai" not in captured_kwargs
    assert body["gemini_status"] == "error"
    assert "403 PERMISSION_DENIED" in body["message"]
    assert "key_type=AI Studio" in body["message"]
    assert aq_secret not in resp.text
    assert body["prediction_result"]["prediction"]["predicted_grade"] >= 0.0

    # 2. When GEMINI_USE_VERTEXAI=true is explicitly set, route to Vertex AI
    monkeypatch.setenv("GEMINI_USE_VERTEXAI", "true")
    resp_vertex = client.post(
        "/recommendations",
        json={
            "subject": "Mathematics",
            "scenario": "early",
            "student": sample_student_payload(include_grades=False),
        },
    )
    assert resp_vertex.status_code == 200
    assert captured_kwargs.get("vertexai") is True
    assert "key_type=Vertex" in resp_vertex.json()["message"]

