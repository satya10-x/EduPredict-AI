# EduPredict AI — System Architecture

This document describes the complete, verified architecture of **EduPredict AI - Intelligent Student Performance Prediction System**, covering the offline training pipeline (Datasets, Data Processing, Classical ML, Deep Learning, and Model Storage) and the online inference and AI advisory pipeline (Streamlit Frontend, FastAPI Backend, and Google Gemini API).

---

## 1. End-to-End System Architecture Diagram

- **Solid arrows (`-->`)** represent the data/training pipeline and **Prediction Flow** (`Streamlit -> FastAPI -> ModelService`).
- **Dashed arrows (`-.->`)** represent the **Gemini Recommendation Flow** (`Streamlit -> FastAPI -> GeminiService -> Google Gemini API`).

```mermaid
flowchart TB
    subgraph L1["1. Dataset Layer (data/raw)"]
        DS_MAT["Mathematics Dataset (student-mat.csv, 395 rows)"]
        DS_POR["Portuguese Dataset (student-por.csv, 649 rows)"]
    end

    subgraph L2["2. Data Processing Layer (notebooks/01-02 & data/processed)"]
        EDA["Data Cleaning & Exploration (01_data_exploration.ipynb, 02_eda_visualization.ipynb)"]
        SPLIT["Train-Test Split (80/20, random_state=42)"]
        PREP["Feature Preprocessing (ColumnTransformer: StandardScaler + OneHotEncoder)"]
    end

    subgraph L3["3. Machine Learning Layer (notebooks/03_machine_learning.ipynb)"]
        ML_ALGOS["Scikit-learn Regressors (Linear Regression, Decision Tree, Random Forest)"]
        SCEN_A["Scenario A: Early Prediction (Excludes G1 & G2, 30 features)"]
        SCEN_B["Scenario B: Late Prediction (Includes G1 & G2, 32 features)"]
    end

    subgraph L4["4. Deep Learning Layer (notebooks/04_deep_learning.ipynb)"]
        DL_NET["TensorFlow/Keras Sequential MLP (Dense 64-32-16-1, ReLU + Linear)"]
        DL_EVAL["EarlyStopping & Test Comparison (artifacts/: MAE, RMSE, R2, Figures)"]
    end

    subgraph L5["5. Model Storage (models/)"]
        STORE_ML["Saved ML Pipelines (math_early/late & portuguese_early/late .joblib)"]
        STORE_DL["Saved Keras Models (*.keras) & Preprocessors (*_nn_preprocessor.joblib)"]
        STORE_META["Model Metadata (ml_model_metadata.json, dl_model_metadata.json)"]
    end

    subgraph L6["6. Backend Layer (backend/)"]
        API["FastAPI Application (backend/main.py, Pydantic schemas.py, CORS)"]
        EP_HEALTH["GET /health (Health & Model Counts)"]
        EP_MODELS["GET /models (Model Catalog & Metrics)"]
        EP_PRED["POST /predict (Prediction API)"]
        EP_REC["POST /recommendations (Gemini Recommendation API)"]
        SVC_MODEL["ModelService (model_service.py: Loads ML & Keras, Raw + Clipped G3)"]
        SVC_GEM["GeminiService (gemini_service.py: Prompt Builder & Error Classifier)"]
    end

    subgraph L7["7. External AI Service"]
        GEMINI["Google Gemini API (google-genai SDK, gemini-3.5-flash-lite, Structured StudyPlan)"]
    end

    subgraph L8["8. Frontend Layer (frontend/app.py + api_client.py)"]
        UI_OVER["Overview Page (Health KPIs & Plotly Charts)"]
        UI_PRED["Student Prediction Page (Early/Late Form & ML/Keras Results)"]
        UI_REC["AI Study Recommendations Page (Personalized Study Plan)"]
        UI_COMP["Model Comparison Page (Metrics Tables & Live Actual vs Predicted)"]
    end

    %% Offline Training & Evaluation Flow
    DS_MAT --> EDA
    DS_POR --> EDA
    EDA --> SPLIT
    SPLIT --> PREP
    PREP --> ML_ALGOS
    ML_ALGOS --> SCEN_A
    ML_ALGOS --> SCEN_B
    PREP --> DL_NET
    SCEN_A --> STORE_ML
    SCEN_B --> STORE_ML
    SCEN_A --> DL_EVAL
    SCEN_B --> DL_EVAL
    DL_NET --> DL_EVAL
    DL_NET --> STORE_DL
    DL_EVAL --> STORE_META

    %% Startup Model Loading
    STORE_ML -->|"Startup Load"| SVC_MODEL
    STORE_DL -->|"Startup Load"| SVC_MODEL
    STORE_META -->|"Startup Load"| SVC_MODEL

    %% Online Prediction Flow (Solid Arrows)
    UI_OVER -->|"GET /health"| API
    UI_COMP -->|"GET /models"| API
    UI_PRED -->|"HTTP POST /predict"| API
    UI_COMP -->|"HTTP POST /predict (Live Eval)"| API
    API --> EP_HEALTH
    API --> EP_MODELS
    API --> EP_PRED
    EP_PRED -->|"Prediction Flow"| SVC_MODEL
    SVC_MODEL -->|"Predicted G3 (Raw & Clipped)"| UI_PRED

    %% Online Gemini Recommendation Flow (Dashed Arrows)
    UI_REC -.->|"HTTP POST /recommendations"| API
    API -.-> EP_REC
    EP_REC -.->|"1. Compute G3 Prediction"| SVC_MODEL
    EP_REC -.->|"2. Send Profile + Predicted G3"| SVC_GEM
    SVC_GEM -.->|"3. HTTPS Request (GEMINI_API_KEY)"| GEMINI
    GEMINI -.->|"4. Structured StudyPlan JSON"| SVC_GEM
    SVC_GEM -.->|"5. Prediction + Study Plan"| UI_REC
```

---

## 2. Runtime Prediction vs. Recommendation Flow

This sequence diagram highlights how **Prediction** (`POST /predict`) works independently offline, while **AI Study Recommendations** (`POST /recommendations`) combines `ModelService` and `Google Gemini API` with graceful degradation if Gemini is unavailable.

```mermaid
sequenceDiagram
    actor Student as User / Student
    participant UI as Streamlit Frontend (frontend/app.py)
    participant API as FastAPI Backend (backend/main.py)
    participant MS as ModelService (models/*.joblib & *.keras)
    participant GS as GeminiService (.env GEMINI_API_KEY)
    participant Gemini as Google Gemini API (AI Studio)

    Note over UI,API: Flow A — Grade Prediction Only (No External AI Call)
    Student->>UI: Submit student features (Subject, Early/Late, ML/Keras)
    UI->>API: POST /predict (JSON validated by Pydantic)
    API->>MS: predict(subject, scenario, student, model_type)
    MS-->>API: PredictionResponse (raw_prediction + clipped predicted_grade)
    API-->>UI: 200 OK (Predicted G3, band, pass/at-risk)
    UI-->>Student: Display predicted marks & ML vs Keras comparison

    Note over UI,Gemini: Flow B — Prediction + Personalized Study Plan
    Student->>UI: Request AI Study Recommendations
    UI->>API: POST /recommendations (JSON)
    API->>MS: predict(...)
    MS-->>API: PredictionResponse
    API->>GS: generate_study_plan(student, prediction)
    GS->>Gemini: generate_content(gemini-3.5-flash-lite, StudyPlan schema)
    alt Gemini API Succeeds
        Gemini-->>GS: Structured JSON StudyPlan
        GS-->>API: ("success", StudyPlan, None)
    else Gemini Unavailable or Error (401/403/429/503/Network)
        GS-->>API: ("unavailable" or "error", None, safe diagnostic message)
    end
    API-->>UI: 200 OK (Always includes ML prediction_result + Gemini status)
    UI-->>Student: Display prediction + AI study plan (or safe warning)
```

---

## 3. Layer-by-Layer Component Breakdown

| Layer | Implementation Files | Purpose & Verified Details |
| :--- | :--- | :--- |
| **1. Dataset Layer** | `data/raw/student-mat.csv`, `data/raw/student-por.csv`, `data/raw/student.txt` | Official UCI Student Performance dataset (ID `320`). Mathematics (`395` rows × `33` cols) and Portuguese (`649` rows × `33` cols), `0` missing values, target `G3` (`0–20`). |
| **2. Data Processing Layer** | `notebooks/01_data_exploration.ipynb`, `02_eda_visualization.ipynb`, `data/processed/*.csv` | Pandas/NumPy exploration, Seaborn/Matplotlib visual EDA, clean `80/20` train-test splits (`random_state=42`), and leakage-free `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`). |
| **3. Machine Learning Layer** | `notebooks/03_machine_learning.ipynb` | Trains `LinearRegression`, `DecisionTreeRegressor`, and `RandomForestRegressor` for all 4 tasks (`Mathematics`/`Portuguese` × `early` without `G1/G2` vs `late` with `G1/G2`), selecting winners strictly on test RMSE. |
| **4. Deep Learning Layer** | `notebooks/04_deep_learning.ipynb`, `artifacts/*.csv`, `artifacts/figures/*.png` | Trains 4 TensorFlow/Keras `Sequential` MLPs (`Dense(64, relu) → Dense(32, relu) → Dense(16, relu) → Dense(1, linear)`) compiled with `Adam` + `MSE` + `MAE`, `EarlyStopping(patience=15)`, and compares test metrics against Day 3 ML models. |
| **5. Model Storage** | `models/*.joblib`, `models/*.keras`, `models/*_metadata.json` | Stores 4 complete Scikit-learn `Pipeline` objects, 4 native `.keras` neural networks, 4 fitted `*_nn_preprocessor.joblib` transformers, and JSON metadata recording winning models, feature lists, and test metrics. |
| **6. Backend Layer** | `backend/main.py`, `schemas.py`, `routes/prediction.py`, `routes/recommendation.py`, `services/model_service.py`, `services/gemini_service.py` | FastAPI REST service with Pydantic v2 validation, CORS for local UI origins, `lifespan` startup model loading, `GET /health`, `GET /models`, `POST /predict`, `POST /recommendations`, and Swagger docs at `/docs`. |
| **7. External AI Service** | Google Gemini API (`google-genai` SDK) | Generates structured `StudyPlan` JSON (`summary`, `strengths`, `risk_factors`, `weekly_study_plan`, `action_items`) using `GEMINI_API_KEY` loaded safely from `.env` via `python-dotenv`, with redacted logging and graceful fallback. |
| **8. Frontend Layer** | `frontend/app.py`, `frontend/api_client.py`, `frontend/sample_data.py` | Four-page Streamlit + Plotly dashboard (`Overview`, `Student Prediction`, `AI Study Recommendations`, `Model Comparison`) communicating strictly via HTTP with `st.session_state` and sample presets. |

