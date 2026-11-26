"""Model loading and inference service for Scikit-learn and optional/lazy Keras models."""

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.schemas import (
    ModelSelectionType,
    ModelTaskInfo,
    PredictionResponse,
    ScenarioType,
    SingleModelPrediction,
    StudentFeatures,
    SubjectType,
)


class KerasDisabledError(RuntimeError):
    """Raised when Keras inference is requested while ENABLE_KERAS_MODELS=false."""


class KerasUnavailableError(RuntimeError):
    """Raised when Keras inference is requested but TensorFlow/Keras is not installed."""


def classify_grade_band(grade: float) -> str:
    """Convert a 0-20 grade into an easy-to-read performance band."""
    if grade >= 16.0:
        return "Excellent (16-20)"
    if grade >= 14.0:
        return "Very Good (14-15)"
    if grade >= 12.0:
        return "Good (12-13)"
    if grade >= 10.0:
        return "Pass (10-11)"
    return "At Risk (<10)"


class ModelService:
    """Loads saved ML pipelines at startup and lazily loads Keras models only when enabled."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.models_dir = self.project_root / "models"
        self.ml_pipelines: dict[tuple[str, str], Any] = {}
        self.ml_meta: dict[tuple[str, str], dict[str, Any]] = {}
        self.keras_models: dict[tuple[str, str], Any] = {}
        self.keras_preprocessors: dict[tuple[str, str], Any] = {}
        self.dl_meta: dict[tuple[str, str], dict[str, Any]] = {}
        self.loaded = False

    def is_keras_enabled(self) -> bool:
        """Return True unless ENABLE_KERAS_MODELS is explicitly disabled."""
        raw = os.getenv("ENABLE_KERAS_MODELS", "true").strip().lower()
        return raw not in {"false", "0", "no", "off"}

    def load_all(self) -> None:
        """
        Load all 4 Scikit-learn pipelines and metadata JSONs at startup.
        TensorFlow/Keras is never imported or loaded here (lazy-loaded on demand).
        """
        if self.loaded:
            return

        ml_meta_path = self.models_dir / "ml_model_metadata.json"
        dl_meta_path = self.models_dir / "dl_model_metadata.json"

        ml_items = json.loads(ml_meta_path.read_text())
        dl_items = json.loads(dl_meta_path.read_text())

        for item in ml_items:
            key = (item["subject"], item["scenario"])
            self.ml_meta[key] = item
            self.ml_pipelines[key] = joblib.load(self.project_root / item["pipeline_file"])

        for item in dl_items:
            key = (item["subject"], item["scenario"])
            self.dl_meta[key] = item

        self.loaded = True

    def ensure_loaded(self) -> None:
        """Lazy-load ML pipelines and metadata if not loaded via FastAPI lifespan startup."""
        if not self.loaded:
            self.load_all()

    def _ensure_keras_task_loaded(self, key: tuple[str, str]) -> None:
        """Lazily import Keras and load the requested .keras model + preprocessor on demand."""
        if not self.is_keras_enabled():
            raise KerasDisabledError(
                "Keras neural network predictions are disabled on this server "
                "(ENABLE_KERAS_MODELS=false). Use model_type='ml' for classical ML predictions."
            )

        if key in self.keras_models and key in self.keras_preprocessors:
            return

        try:
            import keras  # Local lazy import so ML-only deployments never import TensorFlow
        except ImportError as exc:
            raise KerasUnavailableError(
                "TensorFlow/Keras is not installed in this environment. "
                "Use model_type='ml' for classical ML predictions."
            ) from exc

        meta = self.dl_meta[key]
        self.keras_models[key] = keras.models.load_model(self.project_root / meta["model_file"])
        self.keras_preprocessors[key] = joblib.load(self.project_root / meta["preprocessor_file"])

    def _build_input_df(self, student: StudentFeatures, feature_columns: list[str]) -> pd.DataFrame:
        """Build a 1-row DataFrame matching the exact training feature columns."""
        student_dict = student.model_dump()
        row = {col: student_dict[col] for col in feature_columns}
        return pd.DataFrame([row])

    def _format_prediction(self, model_family: str, model_name: str, raw_value: float) -> SingleModelPrediction:
        """Create a prediction object separating raw output from clipped [0, 20] grade."""
        clipped = float(np.clip(raw_value, 0.0, 20.0))
        clipped_rounded = round(clipped, 2)
        return SingleModelPrediction(
            model_family=model_family,
            model_name=model_name,
            raw_prediction=round(float(raw_value), 4),
            predicted_grade=clipped_rounded,
            passed=clipped_rounded >= 10.0,
            performance_band=classify_grade_band(clipped_rounded),
        )

    def _predict_ml(self, subject: SubjectType, scenario: ScenarioType, student: StudentFeatures) -> SingleModelPrediction:
        key = (subject, scenario)
        meta = self.ml_meta[key]
        pipeline = self.ml_pipelines[key]
        input_df = self._build_input_df(student, meta["feature_columns"])
        raw_val = float(pipeline.predict(input_df)[0])
        return self._format_prediction("ml", meta["selected_model"], raw_val)

    def _predict_keras(self, subject: SubjectType, scenario: ScenarioType, student: StudentFeatures) -> SingleModelPrediction:
        key = (subject, scenario)
        self._ensure_keras_task_loaded(key)
        meta = self.dl_meta[key]
        model = self.keras_models[key]
        preprocessor = self.keras_preprocessors[key]
        input_df = self._build_input_df(student, meta["feature_columns"])
        transformed = preprocessor.transform(input_df).astype(np.float32)
        raw_val = float(model.predict(transformed, verbose=0).ravel()[0])
        return self._format_prediction("keras", "Keras Sequential NN", raw_val)

    def predict(
        self,
        subject: SubjectType,
        scenario: ScenarioType,
        student: StudentFeatures,
        model_type: ModelSelectionType = "ml",
    ) -> PredictionResponse:
        """Run prediction using default ML, optional Keras, or both for comparison."""
        self.ensure_loaded()
        key = (subject, scenario)
        default_ml_name = self.ml_meta[key]["selected_model"]

        if model_type == "ml":
            primary = self._predict_ml(subject, scenario, student)
            comparison = None
        elif model_type == "keras":
            primary = self._predict_keras(subject, scenario, student)
            comparison = None
        else:
            ml_pred = self._predict_ml(subject, scenario, student)
            keras_pred = self._predict_keras(subject, scenario, student)
            primary = ml_pred
            comparison = {"ml": ml_pred, "keras": keras_pred}

        return PredictionResponse(
            subject=subject,
            scenario=scenario,
            requested_model_type=model_type,
            default_production_model=default_ml_name,
            prediction=primary,
            comparison=comparison,
        )

    def get_catalog(self) -> list[ModelTaskInfo]:
        """Return metadata and test evaluation metrics for all 4 prediction tasks."""
        self.ensure_loaded()
        tasks: list[ModelTaskInfo] = []
        for key in sorted(self.ml_meta.keys()):
            ml_info = self.ml_meta[key]
            dl_info = self.dl_meta[key]
            tasks.append(
                ModelTaskInfo(
                    subject=ml_info["subject"],
                    scenario=ml_info["scenario"],
                    default_ml_model=ml_info["selected_model"],
                    ml_test_mae=ml_info["test_mae"],
                    ml_test_rmse=ml_info["test_rmse"],
                    ml_test_r2=ml_info["test_r2"],
                    keras_model_name="Keras Sequential NN",
                    keras_test_mae=dl_info["test_mae"],
                    keras_test_rmse=dl_info["test_rmse"],
                    keras_test_r2=dl_info["test_r2"],
                    required_features=ml_info["feature_columns"],
                )
            )
        return tasks
