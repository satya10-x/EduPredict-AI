"""Pydantic request and response schemas for the EduPredict AI API."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator

SubjectType = Literal["Mathematics", "Portuguese"]
ScenarioType = Literal["early", "late"]
ModelSelectionType = Literal["ml", "keras", "both"]


class StudentFeatures(BaseModel):
    """Input attributes matching the UCI Student Performance dataset."""

    school: Literal["GP", "MS"] = Field(..., description="Student's school ('GP' or 'MS').")
    sex: Literal["F", "M"] = Field(..., description="Student's sex ('F' or 'M').")
    age: int = Field(..., ge=15, le=22, description="Student age (15 to 22).")
    address: Literal["U", "R"] = Field(..., description="Home address type ('U' urban, 'R' rural).")
    famsize: Literal["LE3", "GT3"] = Field(..., description="Family size ('LE3' <=3, 'GT3' >3).")
    Pstatus: Literal["T", "A"] = Field(..., description="Parents' cohabitation status ('T' together, 'A' apart).")
    Medu: int = Field(..., ge=0, le=4, description="Mother's education level (0 to 4).")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education level (0 to 4).")
    Mjob: Literal["teacher", "health", "services", "at_home", "other"] = Field(..., description="Mother's job.")
    Fjob: Literal["teacher", "health", "services", "at_home", "other"] = Field(..., description="Father's job.")
    reason: Literal["home", "reputation", "course", "other"] = Field(..., description="Reason for choosing school.")
    guardian: Literal["mother", "father", "other"] = Field(..., description="Student's guardian.")
    traveltime: int = Field(..., ge=1, le=4, description="Home-to-school travel time (1 to 4).")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time category (1 to 4).")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (0 to 4).")
    schoolsup: Literal["yes", "no"] = Field(..., description="Extra educational support ('yes' or 'no').")
    famsup: Literal["yes", "no"] = Field(..., description="Family educational support ('yes' or 'no').")
    paid: Literal["yes", "no"] = Field(..., description="Extra paid classes in the subject ('yes' or 'no').")
    activities: Literal["yes", "no"] = Field(..., description="Extra-curricular activities ('yes' or 'no').")
    nursery: Literal["yes", "no"] = Field(..., description="Attended nursery school ('yes' or 'no').")
    higher: Literal["yes", "no"] = Field(..., description="Wants higher education ('yes' or 'no').")
    internet: Literal["yes", "no"] = Field(..., description="Internet access at home ('yes' or 'no').")
    romantic: Literal["yes", "no"] = Field(..., description="In a romantic relationship ('yes' or 'no').")
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships (1 to 5).")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (1 to 5).")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (1 to 5).")
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (1 to 5).")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (1 to 5).")
    health: int = Field(..., ge=1, le=5, description="Current health status (1 to 5).")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (0 to 93).")

    # Required only for 'late' scenario; ignored in 'early' scenario
    G1: Optional[int] = Field(default=None, ge=0, le=20, description="First-period grade (0 to 20, required for late scenario).")
    G2: Optional[int] = Field(default=None, ge=0, le=20, description="Second-period grade (0 to 20, required for late scenario).")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
                "G1": 13,
                "G2": 14,
            }
        }
    )


class PredictionRequest(BaseModel):
    """Request payload for predicting final grade G3."""

    subject: SubjectType = Field(..., description="Course subject: 'Mathematics' or 'Portuguese'.")
    scenario: ScenarioType = Field(..., description="'early' (no G1/G2) or 'late' (requires G1 and G2).")
    model_type: ModelSelectionType = Field(
        default="ml",
        description="Model family to run: 'ml' (default production model), 'keras', or 'both'.",
    )
    student: StudentFeatures

    @model_validator(mode="after")
    def validate_grades_for_late_scenario(self) -> "PredictionRequest":
        """Require G1 and G2 when scenario is 'late'."""
        if self.scenario == "late":
            if self.student.G1 is None or self.student.G2 is None:
                raise ValueError("Both G1 and G2 grades are required when scenario is 'late'.")
        return self


class SingleModelPrediction(BaseModel):
    """Prediction output from one model family, separating raw vs clipped scores."""

    model_family: Literal["ml", "keras"]
    model_name: str
    raw_prediction: float = Field(..., description="Exact unclipped numeric output from the model.")
    predicted_grade: float = Field(..., description="Display grade clipped to the valid [0.0, 20.0] range.")
    passed: bool = Field(..., description="True when predicted_grade >= 10.0.")
    performance_band: str = Field(..., description="Human-readable performance category.")


class PredictionResponse(BaseModel):
    """Response returned by POST /predict."""

    subject: SubjectType
    scenario: ScenarioType
    requested_model_type: ModelSelectionType
    default_production_model: str
    prediction: SingleModelPrediction
    comparison: Optional[dict[str, SingleModelPrediction]] = None


class StudyPlan(BaseModel):
    """Structured AI study recommendations generated by Google Gemini."""

    summary: str = Field(..., description="Short overall assessment of the student's predicted performance.")
    strengths: list[str] = Field(..., description="Positive academic or study habits observed.")
    risk_factors: list[str] = Field(..., description="Factors that may negatively affect final performance.")
    weekly_study_plan: list[str] = Field(..., description="Concrete weekly study schedule suggestions.")
    action_items: list[str] = Field(..., description="Immediate practical steps for improvement.")


class RecommendationRequest(PredictionRequest):
    """Request payload for POST /recommendations (runs prediction + Gemini study plan)."""


class RecommendationResponse(BaseModel):
    """Response returned by POST /recommendations (always includes ML/DL prediction)."""

    subject: SubjectType
    scenario: ScenarioType
    prediction_result: PredictionResponse
    gemini_status: Literal["success", "unavailable", "error"]
    gemini_model: Optional[str] = None
    recommendations: Optional[StudyPlan] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str
    ml_models_loaded: int
    keras_models_loaded: int
    keras_enabled: bool = True
    gemini_configured: bool


class ModelTaskInfo(BaseModel):
    """Metadata describing one prediction task and its available models."""

    subject: SubjectType
    scenario: ScenarioType
    default_ml_model: str
    ml_test_mae: float
    ml_test_rmse: float
    ml_test_r2: float
    keras_model_name: str
    keras_test_mae: float
    keras_test_rmse: float
    keras_test_r2: float
    required_features: list[str]


class ModelsCatalogResponse(BaseModel):
    """Response returned by GET /models."""

    total_tasks: int
    tasks: list[ModelTaskInfo]

