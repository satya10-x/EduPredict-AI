"""EduPredict AI — Streamlit Frontend Dashboard."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.api_client import (
    check_health,
    fetch_models,
    get_default_backend_url,
    get_recommendations,
    predict_grade,
)
from frontend.sample_data import (
    SAMPLE_STUDENTS,
    load_combined_dataset,
    load_evaluation_data,
    load_unseen_test_rows,
)

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI — Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-card {
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        color: #ffffff;
        margin-bottom: 1.2rem;
    }
    .hero-card h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .hero-card p {
        margin: 0.45rem 0 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pill-pass {
        background-color: #dcfce7;
        color: #166534;
    }
    .pill-risk {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit session_state keys cleanly on first load."""
    if "backend_url" not in st.session_state:
        st.session_state["backend_url"] = get_default_backend_url()
    if "keras_enabled" not in st.session_state:
        st.session_state["keras_enabled"] = True
    if "student_form" not in st.session_state:
        st.session_state["student_form"] = SAMPLE_STUDENTS["Average Student"].copy()
    if "selected_preset" not in st.session_state:
        st.session_state["selected_preset"] = "Average Student"
    if "subject" not in st.session_state:
        st.session_state["subject"] = "Mathematics"
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "early"
    if "model_type" not in st.session_state:
        st.session_state["model_type"] = "ml"
    if "latest_payload" not in st.session_state:
        st.session_state["latest_payload"] = None
    if "latest_prediction" not in st.session_state:
        st.session_state["latest_prediction"] = None
    if "latest_recommendation" not in st.session_state:
        st.session_state["latest_recommendation"] = None
    if "live_actual_vs_pred" not in st.session_state:
        st.session_state["live_actual_vs_pred"] = {}


def apply_preset(preset_name: str) -> None:
    """Populate session_state student fields from a named sample profile."""
    if preset_name in SAMPLE_STUDENTS:
        st.session_state["student_form"] = SAMPLE_STUDENTS[preset_name].copy()
        st.session_state["selected_preset"] = preset_name


def build_payload(
    subject: str,
    scenario: str,
    model_type: str,
    student_data: dict,
) -> dict:
    """Construct a valid JSON request payload for FastAPI /predict and /recommendations."""
    student_payload = student_data.copy()
    if scenario == "early":
        student_payload.pop("G1", None)
        student_payload.pop("G2", None)
    return {
        "subject": subject,
        "scenario": scenario,
        "model_type": model_type,
        "student": student_payload,
    }


def validate_student_inputs(student_data: dict, scenario: str) -> list[str]:
    """Perform basic client-side validation before sending HTTP requests."""
    errors = []
    if not (15 <= int(student_data["age"]) <= 22):
        errors.append("Age must be between 15 and 22.")
    if not (0 <= int(student_data["absences"]) <= 93):
        errors.append("Absences must be between 0 and 93.")
    if scenario == "late":
        for grade_key in ["G1", "G2"]:
            val = student_data.get(grade_key)
            if val is None or not (0 <= int(val) <= 20):
                errors.append(f"{grade_key} must be between 0 and 20 for Late prediction.")
    return errors


def render_prediction_cards(pred_response: dict) -> None:
    """Display prediction metrics, band badge, and optional ML vs Keras comparison."""
    primary = pred_response["prediction"]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Predicted Final Grade (G3)",
        value=f"{primary['predicted_grade']:.2f} / 20",
        delta=f"Raw: {primary['raw_prediction']:.2f}",
    )
    col2.metric(
        label="Performance Band",
        value=primary["performance_band"],
    )
    col3.metric(
        label="Selected Model",
        value=primary["model_name"],
        delta=f"Family: {primary['model_family'].upper()}",
    )
    col4.metric(
        label="Scenario & Subject",
        value=f"{pred_response['subject']}",
        delta=f"{pred_response['scenario'].title()} Prediction",
    )

    if primary["passed"]:
        st.markdown(
            '<span class="status-pill pill-pass">Projected Outcome: PASS (G3 >= 10)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill pill-risk">Projected Outcome: AT RISK (G3 < 10)</span>',
            unsafe_allow_html=True,
        )

    comparison = pred_response.get("comparison")
    if comparison:
        st.markdown("#### ML vs Keras Neural Network Comparison")
        comp_rows = [
            {
                "Model Family": "Scikit-learn (Default ML)",
                "Model Name": comparison["ml"]["model_name"],
                "Predicted Grade (Clipped)": comparison["ml"]["predicted_grade"],
                "Raw Output": comparison["ml"]["raw_prediction"],
                "Performance Band": comparison["ml"]["performance_band"],
            },
            {
                "Model Family": "TensorFlow / Keras",
                "Model Name": comparison["keras"]["model_name"],
                "Predicted Grade (Clipped)": comparison["keras"]["predicted_grade"],
                "Raw Output": comparison["keras"]["raw_prediction"],
                "Performance Band": comparison["keras"]["performance_band"],
            },
        ]
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        fig_cmp = px.bar(
            comp_df,
            x="Model Family",
            y="Predicted Grade (Clipped)",
            color="Model Family",
            text="Predicted Grade (Clipped)",
            range_y=[0, 20.5],
            title="Predicted Final Grade Comparison (ML vs Keras)",
        )
        fig_cmp.add_hline(y=10, line_dash="dot", line_color="crimson", annotation_text="Pass (10)")
        fig_cmp.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ------------------------------------------------------------------------------
# Sidebar Navigation & Live Health Check
# ------------------------------------------------------------------------------
def render_sidebar() -> str:
    """Render sidebar navigation and live FastAPI health indicator."""
    with st.sidebar:
        st.title("🎓 EduPredict AI")
        st.caption("Intelligent Student Performance Prediction System")
        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "Overview",
                "Student Prediction",
                "AI Study Recommendations",
                "Model Comparison",
            ],
        )

        st.divider()
        st.subheader("Backend Connection")
        backend_url = st.text_input(
            "FastAPI Server URL",
            value=st.session_state["backend_url"],
            help="Address where uvicorn backend.main:app is running.",
        ).rstrip("/")
        st.session_state["backend_url"] = backend_url

        health = check_health(backend_url)
        if health.ok and health.data:
            keras_enabled = health.data.get("keras_enabled", True)
            st.session_state["keras_enabled"] = keras_enabled
            st.success("Backend Online")
            keras_label = (
                f"Keras Models: **{health.data.get('keras_models_loaded', 0)}**"
                if keras_enabled
                else "Mode: **ML-Only**"
            )
            st.caption(
                f"ML Models: **{health.data.get('ml_models_loaded', 0)}** | {keras_label}"
            )
            gemini_ok = health.data.get("gemini_configured", False)
            st.caption(f"Gemini API: **{'Configured' if gemini_ok else 'Not Configured'}**")
        else:
            st.error("Backend Offline")
            st.caption(health.error or "Start backend: `uvicorn backend.main:app --reload`")

    return page


# ------------------------------------------------------------------------------
# Page 1: Overview
# ------------------------------------------------------------------------------
def render_overview_page(backend_url: str) -> None:
    """Render the Overview dashboard page."""
    st.markdown(
        """
        <div class="hero-card">
            <h1>🎓 EduPredict AI</h1>
            <p>Intelligent Student Performance Prediction &amp; Personalized Study Planning powered by Scikit-learn, TensorFlow/Keras, FastAPI, and Google Gemini.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    health = check_health(backend_url)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("UCI Math Records", "395", "33 Attributes")
    col2.metric("UCI Portuguese Records", "649", "33 Attributes")
    col3.metric("Prediction Scenarios", "4 Tasks", "Early & Late")
    if health.ok and health.data:
        mode_note = (
            f"{health.data['ml_models_loaded']} ML + {health.data['keras_models_loaded']} Keras"
            if health.data.get("keras_enabled", True)
            else f"{health.data['ml_models_loaded']} ML (ML-Only Mode)"
        )
        col4.metric(
            "FastAPI Status",
            "Online",
            mode_note,
        )
    else:
        col4.metric("FastAPI Status", "Offline", "Waking up or not running")

    st.markdown("### System Architecture")
    a_col1, a_col2, a_col3 = st.columns(3)
    with a_col1:
        st.info(
            "**1. ML & Deep Learning Engine**\n\n"
            "Trained separate Scikit-learn pipelines and Keras Sequential neural networks "
            "for Mathematics and Portuguese in **Early** (no G1/G2) and **Late** (with G1/G2) modes."
        )
    with a_col2:
        st.info(
            "**2. FastAPI REST Backend**\n\n"
            "Validates student inputs with Pydantic, serves predictions at `/predict`, "
            "exposes model metrics at `/models`, and integrates Gemini at `/recommendations`."
        )
    with a_col3:
        st.info(
            "**3. Interactive Streamlit UI**\n\n"
            "Communicates strictly via HTTP requests without duplicating ML logic or "
            "exposing API keys in the browser."
        )

    # Interactive Plotly visualizations
    combined_df = load_combined_dataset()
    eval_tables = load_evaluation_data()

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        if combined_df is not None:
            fig_dist = px.histogram(
                combined_df,
                x="G3",
                color="subject",
                barmode="overlay",
                nbins=21,
                opacity=0.7,
                title="Final Grade (G3) Distribution Across Subjects",
                labels={"G3": "Final Grade (0-20)", "subject": "Subject"},
            )
            fig_dist.add_vline(x=10, line_dash="dot", line_color="crimson", annotation_text="Pass (10)")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("Processed dataset `data/processed/student_combined_clean.csv` not found.")

    with v_col2:
        if "comparison" in eval_tables:
            cmp_df = eval_tables["comparison"].copy()
            cmp_df["Task"] = cmp_df["Subject"] + " (" + cmp_df["Scenario"].str.title() + ")"
            melted_r2 = cmp_df.melt(
                id_vars="Task",
                value_vars=["ML_Test_R2", "DL_Test_R2"],
                var_name="Model Family",
                value_name="Test R2",
            ).replace({"ML_Test_R2": "Best Scikit-learn (Day 3)", "DL_Test_R2": "Keras NN (Day 4)"})
            fig_r2 = px.bar(
                melted_r2,
                x="Task",
                y="Test R2",
                color="Model Family",
                barmode="group",
                text="Test R2",
                title="Early vs Late Generalization (Test R2 Score)",
            )
            fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            st.plotly_chart(fig_r2, use_container_width=True)
        else:
            st.warning("Evaluation summary `artifacts/dl_vs_ml_comparison.csv` not found.")


# ------------------------------------------------------------------------------
# Page 2: Student Prediction
# ------------------------------------------------------------------------------
def render_prediction_page(backend_url: str) -> None:
    """Render the Student Prediction input form and live FastAPI prediction display."""
    st.header("🎯 Student Performance Prediction")
    st.write("Configure the subject, prediction scenario, and student details to predict final grade (`G3`).")

    # Preset bar (updates session_state cleanly)
    p_col1, p_col2 = st.columns([3, 1])
    with p_col1:
        chosen_preset = st.selectbox(
            "Load Sample Student Profile for Demonstration",
            options=list(SAMPLE_STUDENTS.keys()),
            index=list(SAMPLE_STUDENTS.keys()).index(st.session_state["selected_preset"]),
        )
    with p_col2:
        st.write("")
        st.write("")
        if st.button("Apply Sample Profile", use_container_width=True):
            apply_preset(chosen_preset)
            st.rerun()

    # Scenario & Model Controls
    c_col1, c_col2, c_col3 = st.columns(3)
    with c_col1:
        subject = st.selectbox(
            "Subject",
            options=["Mathematics", "Portuguese"],
            index=0 if st.session_state["subject"] == "Mathematics" else 1,
        )
    with c_col2:
        scenario_label = st.selectbox(
            "Prediction Scenario",
            options=[
                "early — Before term grades (excludes G1 & G2)",
                "late — Mid-year prediction (requires G1 & G2)",
            ],
            index=0 if st.session_state["scenario"] == "early" else 1,
        )
        scenario = "early" if scenario_label.startswith("early") else "late"
    with c_col3:
        model_label = st.selectbox(
            "Model Selection",
            options=[
                "ml — Best Production ML Model (Default)",
                "keras — TensorFlow/Keras Sequential NN",
                "both — Compare ML vs Keras Side-by-Side",
            ],
            index=["ml", "keras", "both"].index(st.session_state["model_type"]),
        )
        model_type = model_label.split(" — ")[0]

    st.session_state["subject"] = subject
    st.session_state["scenario"] = scenario
    st.session_state["model_type"] = model_type

    if not st.session_state.get("keras_enabled", True) and model_type in ("keras", "both"):
        st.info(
            "Connected backend is running in ML-only mode (`ENABLE_KERAS_MODELS=false`). "
            "Switch Model Selection to `ml` for classical ML predictions."
        )

    s = st.session_state["student_form"]

    with st.form("student_prediction_form"):
        st.subheader("Student Academic & Background Information")

        tab1, tab2, tab3 = st.tabs(["Academic & Grades", "Family & Background", "Study Habits & Lifestyle"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            school = col1.selectbox("School", ["GP", "MS"], index=["GP", "MS"].index(s["school"]))
            studytime = col2.selectbox(
                "Weekly Study Time (1:<2h, 2:2-5h, 3:5-10h, 4:>10h)",
                [1, 2, 3, 4],
                index=[1, 2, 3, 4].index(s["studytime"]),
            )
            failures = col3.number_input("Past Class Failures (0-4)", min_value=0, max_value=4, value=int(s["failures"]))

            col4, col5, col6 = st.columns(3)
            absences = col4.number_input("School Absences (0-93)", min_value=0, max_value=93, value=int(s["absences"]))
            schoolsup = col5.selectbox("Extra School Support", ["yes", "no"], index=["yes", "no"].index(s["schoolsup"]))
            famsup = col6.selectbox("Family Educational Support", ["yes", "no"], index=["yes", "no"].index(s["famsup"]))

            col7, col8, col9 = st.columns(3)
            paid = col7.selectbox("Extra Paid Classes", ["yes", "no"], index=["yes", "no"].index(s["paid"]))
            higher = col8.selectbox("Wants Higher Education", ["yes", "no"], index=["yes", "no"].index(s["higher"]))
            reason = col9.selectbox(
                "Reason for Choosing School",
                ["course", "home", "reputation", "other"],
                index=["course", "home", "reputation", "other"].index(s["reason"]),
            )

            if scenario == "late":
                st.markdown("**Prior Period Grades (Required for Late Prediction)**")
                g_col1, g_col2 = st.columns(2)
                g1_val = g_col1.number_input("First Period Grade G1 (0-20)", min_value=0, max_value=20, value=int(s.get("G1", 11)))
                g2_val = g_col2.number_input("Second Period Grade G2 (0-20)", min_value=0, max_value=20, value=int(s.get("G2", 11)))
            else:
                g1_val = s.get("G1", 11)
                g2_val = s.get("G2", 11)
                st.info("Early Prediction mode active: `G1` and `G2` are excluded from features.")

        with tab2:
            col1, col2, col3 = st.columns(3)
            sex = col1.selectbox("Sex", ["F", "M"], index=["F", "M"].index(s["sex"]))
            age = col2.number_input("Age (15-22)", min_value=15, max_value=22, value=int(s["age"]))
            address = col3.selectbox("Home Address ('U' Urban, 'R' Rural)", ["U", "R"], index=["U", "R"].index(s["address"]))

            col4, col5, col6 = st.columns(3)
            famsize = col4.selectbox("Family Size ('GT3' >3, 'LE3' <=3)", ["GT3", "LE3"], index=["GT3", "LE3"].index(s["famsize"]))
            pstatus = col5.selectbox("Parent Cohabitation ('T' Together, 'A' Apart)", ["T", "A"], index=["T", "A"].index(s["Pstatus"]))
            guardian = col6.selectbox("Primary Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(s["guardian"]))

            col7, col8, col9 = st.columns(3)
            medu = col7.selectbox("Mother's Education (0-4)", [0, 1, 2, 3, 4], index=int(s["Medu"]))
            fedu = col8.selectbox("Father's Education (0-4)", [0, 1, 2, 3, 4], index=int(s["Fedu"]))
            famrel = col9.selectbox("Family Relationship Quality (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["famrel"]))

            col10, col11 = st.columns(2)
            mjob = col10.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(s["Mjob"]))
            fjob = col11.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(s["Fjob"]))

        with tab3:
            col1, col2, col3 = st.columns(3)
            traveltime = col1.selectbox("Travel Time (1:<15m .. 4:>1h)", [1, 2, 3, 4], index=[1, 2, 3, 4].index(s["traveltime"]))
            activities = col2.selectbox("Extra-Curricular Activities", ["yes", "no"], index=["yes", "no"].index(s["activities"]))
            nursery = col3.selectbox("Attended Nursery School", ["yes", "no"], index=["yes", "no"].index(s["nursery"]))

            col4, col5, col6 = st.columns(3)
            internet = col4.selectbox("Internet Access at Home", ["yes", "no"], index=["yes", "no"].index(s["internet"]))
            romantic = col5.selectbox("In a Romantic Relationship", ["yes", "no"], index=["yes", "no"].index(s["romantic"]))
            freetime = col6.selectbox("Free Time After School (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["freetime"]))

            col7, col8, col9, col10 = st.columns(4)
            goout = col7.selectbox("Going Out with Friends (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["goout"]))
            dalc = col8.selectbox("Workday Alcohol (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["Dalc"]))
            walc = col9.selectbox("Weekend Alcohol (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["Walc"]))
            health = col10.selectbox("Health Status (1-5)", [1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5].index(s["health"]))

        submitted = st.form_submit_button("Predict Final Grade (G3)", type="primary", use_container_width=True)

    updated_student = {
        "school": school,
        "sex": sex,
        "age": int(age),
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": int(medu),
        "Fedu": int(fedu),
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": int(traveltime),
        "studytime": int(studytime),
        "failures": int(failures),
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": int(famrel),
        "freetime": int(freetime),
        "goout": int(goout),
        "Dalc": int(dalc),
        "Walc": int(walc),
        "health": int(health),
        "absences": int(absences),
        "G1": int(g1_val),
        "G2": int(g2_val),
    }

    if submitted:
        st.session_state["student_form"] = updated_student
        val_errors = validate_student_inputs(updated_student, scenario)
        if val_errors:
            for err in val_errors:
                st.error(err)
            return

        payload = build_payload(subject, scenario, model_type, updated_student)
        st.session_state["latest_payload"] = payload

        with st.spinner("Calling FastAPI /predict..."):
            res = predict_grade(payload, base_url=backend_url)

        if res.ok and res.data:
            st.session_state["latest_prediction"] = res.data
            st.success("Prediction received from FastAPI backend!")
        else:
            st.error(res.error or "Prediction failed.")

    if st.session_state.get("latest_prediction"):
        st.divider()
        st.subheader("Latest Prediction Result")
        render_prediction_cards(st.session_state["latest_prediction"])


# ------------------------------------------------------------------------------
# Page 3: AI Study Recommendations
# ------------------------------------------------------------------------------
def render_recommendations_page(backend_url: str) -> None:
    """Render the Gemini-powered AI Study Recommendations page."""
    st.header("🤖 AI Study Recommendations (Google Gemini)")
    st.write(
        "Combine FastAPI's grade prediction with **Google Gemini** to generate a tailored, "
        "actionable study plan for the active student profile."
    )

    if st.session_state.get("latest_payload"):
        active_payload = st.session_state["latest_payload"]
        st.info(
            f"Using active student profile: **{active_payload['subject']}** | "
            f"**{active_payload['scenario'].title()}** prediction | "
            f"Preset/Edited form ready."
        )
    else:
        st.info("No prediction run yet — using the current preset/form values from `session_state`.")
        active_payload = build_payload(
            st.session_state["subject"],
            st.session_state["scenario"],
            st.session_state["model_type"],
            st.session_state["student_form"],
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        preset_override = st.selectbox(
            "Or quickly switch student profile",
            options=["Keep Current Profile"] + list(SAMPLE_STUDENTS.keys()),
        )
    with c2:
        st.write("")
        st.write("")
        if preset_override != "Keep Current Profile":
            apply_preset(preset_override)
            active_payload = build_payload(
                st.session_state["subject"],
                st.session_state["scenario"],
                st.session_state["model_type"],
                st.session_state["student_form"],
            )

    if st.button("Generate AI Study Plan", type="primary", use_container_width=True):
        with st.spinner("Running prediction and requesting Gemini study plan..."):
            res = get_recommendations(active_payload, base_url=backend_url)

        if res.ok and res.data:
            st.session_state["latest_recommendation"] = res.data
            st.session_state["latest_prediction"] = res.data["prediction_result"]
        else:
            st.error(res.error or "Failed to reach `/recommendations`.")

    rec_data = st.session_state.get("latest_recommendation")
    if rec_data:
        st.divider()
        st.subheader("Predicted Academic Outlook")
        render_prediction_cards(rec_data["prediction_result"])

        st.divider()
        st.subheader("Personalized Study Plan")
        status = rec_data.get("gemini_status")
        plan = rec_data.get("recommendations")

        if status == "success" and plan:
            if rec_data.get("gemini_model"):
                st.caption(f"Generated by Google Gemini (`{rec_data['gemini_model']}`)")
            st.success(plan["summary"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Strengths")
                for item in plan["strengths"]:
                    st.markdown(f"- {item}")

                st.markdown("#### Weekly Study Plan")
                for item in plan["weekly_study_plan"]:
                    st.markdown(f"- {item}")

            with col_b:
                st.markdown("#### Risk Factors")
                for item in plan["risk_factors"]:
                    st.markdown(f"- {item}")

                st.markdown("#### Immediate Action Items")
                for item in plan["action_items"]:
                    st.markdown(f"- {item}")
        else:
            st.warning(
                f"Gemini Study Plan Unavailable (`{status}`): "
                f"{rec_data.get('message', 'Gemini API could not be reached.')} "
                "Note that the ML prediction above still succeeded."
            )


# ------------------------------------------------------------------------------
# Page 4: Model Comparison
# ------------------------------------------------------------------------------
def render_model_comparison_page(backend_url: str) -> None:
    """Render evaluation metrics (MAE, RMSE, R2), live /models catalog, and Plotly charts."""
    st.header("📊 Model Comparison & Evaluation Results")

    # 1. Live FastAPI /models catalog
    st.subheader("1. Registered Models in FastAPI (`GET /models`)")
    catalog_res = fetch_models(backend_url)
    if catalog_res.ok and catalog_res.data:
        tasks_df = pd.DataFrame(catalog_res.data["tasks"])
        display_cols = [
            "subject",
            "scenario",
            "default_ml_model",
            "ml_test_mae",
            "ml_test_rmse",
            "ml_test_r2",
            "keras_model_name",
            "keras_test_mae",
            "keras_test_rmse",
            "keras_test_r2",
        ]
        st.dataframe(tasks_df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.warning(catalog_res.error or "Backend offline — start FastAPI to view live `/models` registry.")

    # 2. Full Day 3 + Day 4 Evaluation Tables & Plotly Model Comparison Chart
    eval_tables = load_evaluation_data()
    if "ml" in eval_tables and "dl" in eval_tables:
        ml_df = eval_tables["ml"].copy()
        dl_df = eval_tables["dl"].copy()
        dl_df["Model"] = "Keras Sequential NN"

        combined_eval = pd.concat(
            [
                ml_df[["Subject", "Scenario", "Model", "Test_MAE", "Test_RMSE", "Test_R2"]],
                dl_df[["Subject", "Scenario", "Model", "Test_MAE", "Test_RMSE", "Test_R2"]],
            ],
            ignore_index=True,
        )
        combined_eval["Task"] = combined_eval["Subject"] + " (" + combined_eval["Scenario"].str.title() + ")"

        st.subheader("2. Unseen Test Performance Comparison (MAE, RMSE, R2)")
        metric_choice = st.selectbox(
            "Select Evaluation Metric to Plot",
            options=["Test_RMSE", "Test_MAE", "Test_R2"],
            index=0,
        )

        fig_models = px.bar(
            combined_eval,
            x="Task",
            y=metric_choice,
            color="Model",
            barmode="group",
            text=metric_choice,
            title=f"All Models Across All 4 Tasks — {metric_choice}",
        )
        fig_models.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        if metric_choice == "Test_R2":
            fig_models.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_models, use_container_width=True)

        with st.expander("View Complete Day 3 (ML) & Day 4 (Keras) Evaluation Tables"):
            st.markdown("**Day 3 — Classical ML Evaluation (`ml_evaluation_results.csv`)**")
            st.dataframe(ml_df, use_container_width=True, hide_index=True)
            st.markdown("**Day 4 — Keras Neural Network Evaluation (`dl_evaluation_results.csv`)**")
            st.dataframe(dl_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Evaluation CSV files not found in `artifacts/`.")

    # 3. Live Actual vs Predicted Plotly Chart via POST /predict
    st.subheader("3. Live Actual vs Predicted Final Grades (Unseen Test Samples via `POST /predict`)")
    st.write(
        "This chart sends real unseen test student records to the FastAPI `/predict` endpoint "
        "and plots actual `G3` vs live predicted `G3` without duplicating any ML logic in the frontend."
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1, 1, 1.3])
    eval_subj = f_col1.selectbox("Subject", ["Mathematics", "Portuguese"], key="cmp_subj")
    eval_scen = f_col2.selectbox("Scenario", ["late", "early"], key="cmp_scen")
    eval_mtype = f_col3.selectbox("Model", ["ml", "keras"], key="cmp_mtype")
    n_samples = f_col4.slider("Unseen Test Samples", min_value=10, max_value=40, value=20, step=5)

    if st.button("Run Live Actual vs Predicted Evaluation", type="primary"):
        loaded = load_unseen_test_rows(eval_subj, limit=n_samples)
        if loaded is None:
            st.error("Raw dataset files not found in `data/raw/`.")
        else:
            X_sample, y_sample = loaded
            rows_out = []
            failed_msg = None
            with st.spinner(f"Querying FastAPI `/predict` for {len(X_sample)} unseen test students..."):
                for (_, row), actual_g3 in zip(X_sample.iterrows(), y_sample.values):
                    payload = build_payload(eval_subj, eval_scen, eval_mtype, row.to_dict())
                    res = predict_grade(payload, base_url=backend_url)
                    if not res.ok or not res.data:
                        failed_msg = res.error or "Request failed."
                        break
                    pred_info = res.data["prediction"]
                    rows_out.append({
                        "Actual_G3": float(actual_g3),
                        "Predicted_G3": float(pred_info["predicted_grade"]),
                        "Abs_Error": abs(float(actual_g3) - float(pred_info["predicted_grade"])),
                        "Model": pred_info["model_name"],
                    })

            if failed_msg:
                st.error(failed_msg)
            else:
                st.session_state["live_actual_vs_pred"][(eval_subj, eval_scen, eval_mtype)] = pd.DataFrame(rows_out)

    cached_df = st.session_state["live_actual_vs_pred"].get((eval_subj, eval_scen, eval_mtype))
    if cached_df is not None and not cached_df.empty:
        sample_mae = float(cached_df["Abs_Error"].mean())
        sample_rmse = float(np.sqrt(np.mean((cached_df["Actual_G3"] - cached_df["Predicted_G3"]) ** 2)))
        m1, m2, m3 = st.columns(3)
        m1.metric("Samples Evaluated", len(cached_df))
        m2.metric("Sample MAE", f"{sample_mae:.2f}")
        m3.metric("Sample RMSE", f"{sample_rmse:.2f}")

        fig_ap = px.scatter(
            cached_df,
            x="Actual_G3",
            y="Predicted_G3",
            hover_data=["Abs_Error", "Model"],
            title=f"Live Actual vs Predicted G3 — {eval_subj} ({eval_scen.title()}, {cached_df['Model'].iloc[0]})",
            labels={"Actual_G3": "Actual Final Grade (G3)", "Predicted_G3": "Predicted Final Grade (G3)"},
        )
        fig_ap.add_trace(
            go.Scatter(
                x=[0, 20],
                y=[0, 20],
                mode="lines",
                name="Ideal (Actual = Predicted)",
                line=dict(color="crimson", dash="dash"),
            )
        )
        fig_ap.update_xaxes(range=[-0.5, 20.5])
        fig_ap.update_yaxes(range=[-0.5, 20.5])
        st.plotly_chart(fig_ap, use_container_width=True)


# ------------------------------------------------------------------------------
# Main Application Entry
# ------------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    page = render_sidebar()
    backend_url = st.session_state["backend_url"]

    if page == "Overview":
        render_overview_page(backend_url)
    elif page == "Student Prediction":
        render_prediction_page(backend_url)
    elif page == "AI Study Recommendations":
        render_recommendations_page(backend_url)
    else:
        render_model_comparison_page(backend_url)


if __name__ == "__main__":
    main()

