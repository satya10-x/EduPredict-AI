# EduPredict AI — FastAPI Backend

Modular REST API serving trained **Scikit-learn** pipelines and **TensorFlow/Keras** neural networks for student final grade (`G3`, scale `0–20`) prediction, integrated with the **Google Gemini API** (`google-genai`) for personalized study recommendations.

---

## Folder Structure

```text
backend/
├── main.py                      # FastAPI app, CORS, startup model loading, GET /health
├── schemas.py                   # Pydantic v2 validation schemas & error rules
├── routes/
│   ├── prediction.py            # GET /models, POST /predict
│   └── recommendation.py        # POST /recommendations
├── services/
│   ├── model_service.py         # Loads saved ML (.joblib) & Keras (.keras) models read-only
│   └── gemini_service.py        # Reads GEMINI_API_KEY from .env and calls google-genai safely
├── tests/
│   └── test_api.py              # Automated pytest suite
└── README.md                    # This documentation
```

---

## Running the Backend

1. **Activate the virtual environment** (from the project root):
   ```bash
   source venv/bin/activate
   ```
2. **Configure environment variables** (optional for ML-only predictions, required for live Gemini study plans):
   ```bash
   cp .env.example .env
   # Add your real GEMINI_API_KEY inside .env (never commit .env)
   ```
3. **Start the FastAPI server**:
   ```bash
   uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
4. **Open Interactive Swagger Documentation**:
   - Swagger UI: `http://127.0.0.1:8000/docs`
   - ReDoc: `http://127.0.0.1:8000/redoc`

---

## API Endpoints

### 1. `GET /health`
Checks server health, loaded ML/Keras model counts, and whether `GEMINI_API_KEY` is configured (boolean only—never exposes keys).

**Example Response (`200 OK`):**
```json
{
  "status": "healthy",
  "ml_models_loaded": 4,
  "keras_models_loaded": 4,
  "gemini_configured": true
}
```

---

### 2. `GET /models`
Lists all 4 prediction tasks (`Mathematics` / `Portuguese` × `early` / `late`), their default classical ML production model, Keras model metrics, and required input feature names.

**Example Response (`200 OK`, truncated to 1 task):**
```json
{
  "total_tasks": 4,
  "tasks": [
    {
      "subject": "Mathematics",
      "scenario": "early",
      "default_ml_model": "Random Forest Regressor",
      "ml_test_mae": 2.9938,
      "ml_test_rmse": 3.7508,
      "ml_test_r2": 0.3139,
      "keras_model_name": "Keras Sequential NN",
      "keras_test_mae": 3.4899,
      "keras_test_rmse": 4.4397,
      "keras_test_r2": 0.0387,
      "required_features": ["school", "sex", "age", "..."]
    }
  ]
}
```

---

### 3. `POST /predict`
Predicts final grade `G3` (`0–20`).
- Uses the **best Day 3 Scikit-learn pipeline by default** (`"model_type": "ml"`).
- Set `"model_type": "keras"` to query the Keras Neural Network, or `"model_type": "both"` to compare ML and Keras side by side.
- Returns **both** `raw_prediction` (exact unclipped model output) and `predicted_grade` (clipped to `[0.0, 20.0]`).
- For `"scenario": "late"`, both `G1` and `G2` must be provided (`422 Unprocessable Entity` if missing). For `"scenario": "early"`, `G1` and `G2` may be omitted and are never passed to the early models.

**Example Request (`POST /predict` — Late Prediction & Comparison):**
```json
{
  "subject": "Portuguese",
  "scenario": "late",
  "model_type": "both",
  "student": {
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
    "G2": 14
  }
}
```

**Example Response (`200 OK`):**
```json
{
  "subject": "Portuguese",
  "scenario": "late",
  "requested_model_type": "both",
  "default_production_model": "Linear Regression",
  "prediction": {
    "model_family": "ml",
    "model_name": "Linear Regression",
    "raw_prediction": 14.3215,
    "predicted_grade": 14.32,
    "passed": true,
    "performance_band": "Very Good (14-15)"
  },
  "comparison": {
    "ml": {
      "model_family": "ml",
      "model_name": "Linear Regression",
      "raw_prediction": 14.3215,
      "predicted_grade": 14.32,
      "passed": true,
      "performance_band": "Very Good (14-15)"
    },
    "keras": {
      "model_family": "keras",
      "model_name": "Keras Sequential NN",
      "raw_prediction": 14.0842,
      "predicted_grade": 14.08,
      "passed": true,
      "performance_band": "Very Good (14-15)"
    }
  }
}
```

---

### 4. `POST /recommendations`
Accepts the same request body as `POST /predict`, computes the grade prediction, and asks **Google Gemini** to generate a structured study plan (`summary`, `strengths`, `risk_factors`, `weekly_study_plan`, `action_items`).

If Gemini is unavailable or `GEMINI_API_KEY` is not set, `prediction_result` is **still returned** (`200 OK`) with `gemini_status: "unavailable"` (or `"error"`) and `recommendations: null`.

**Example Response (`200 OK`, Gemini Available):**
```json
{
  "subject": "Mathematics",
  "scenario": "early",
  "prediction_result": { "...": "..." },
  "gemini_status": "success",
  "gemini_model": "gemini-2.5-flash",
  "recommendations": {
    "summary": "Predicted score shows solid potential with room to strengthen problem-solving habits.",
    "strengths": ["Strong family support", "Low workday alcohol consumption"],
    "risk_factors": ["Moderate absences (4)"],
    "weekly_study_plan": ["3 focused 45-minute practice blocks per week"],
    "action_items": ["Review missed class notes within 24 hours"]
  },
  "message": null
}
```

---

## Running Backend Tests

```bash
pytest backend/tests -v
```

