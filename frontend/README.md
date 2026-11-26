# EduPredict AI — Streamlit Frontend

Interactive web dashboard built with **Streamlit**, **Plotly**, **Pandas**, and **Requests** that connects to the **FastAPI** backend over HTTP.

---

## Pages Included

1. **Overview:** Project summary, live backend/model health status (`GET /health`), grade distributions, and Early vs Late performance charts.
2. **Student Prediction:** Sample student presets, Subject (`Mathematics` / `Portuguese`), Scenario (`early` / `late`), Model Selection (`ml`, `keras`, `both`), full UCI input validation, and `POST /predict` results.
3. **AI Study Recommendations:** Combines the active student prediction with Google Gemini study plans via `POST /recommendations`, with graceful warnings if Gemini is unavailable.
4. **Model Comparison:** Live `/models` registry, Day 3 vs Day 4 `MAE`, `RMSE`, and `R²` evaluation tables, interactive Plotly comparison charts, and live **Actual vs Predicted** evaluation via `POST /predict`.

---

## How to Run (Two Terminals)

Always keep the **FastAPI backend** and **Streamlit frontend** running in separate terminals from the project root.

### Terminal 1 — Start FastAPI Backend
```bash
source venv/bin/activate
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
- Swagger API Docs: `http://127.0.0.1:8000/docs`

### Terminal 2 — Start Streamlit Frontend
```bash
source venv/bin/activate
streamlit run frontend/app.py --server.port 8501
```
- Dashboard URL: `http://127.0.0.1:8501`

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (`https://github.com/satya10-x/EduPredict-AI`).
2. In **Streamlit Community Cloud**, set:
   - **Repository:** `satya10-x/EduPredict-AI`
   - **Branch:** `main`
   - **Main file path:** `frontend/app.py`
3. In **App Settings → Secrets**, add only the backend URL (never add `GEMINI_API_KEY` to the frontend):
   ```toml
   BACKEND_URL = "https://edupredict-ai-vfnx.onrender.com"
   ```
4. Once deployed, copy your `https://<your-app>.streamlit.app` URL into your Render backend's `ALLOWED_ORIGINS` environment variable.

