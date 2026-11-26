"""Sample student profiles and read-only dataset/evaluation loaders for Streamlit."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

SAMPLE_STUDENTS: dict[str, dict] = {
    "High-Achieving Student": {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "teacher",
        "Fjob": "services",
        "reason": "reputation",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "famsup": "yes",
        "paid": "yes",
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
        "health": 5,
        "absences": 2,
        "G1": 16,
        "G2": 17,
    },
    "Average Student": {
        "school": "GP",
        "sex": "M",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "T",
        "Medu": 3,
        "Fedu": 2,
        "Mjob": "services",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
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
        "goout": 3,
        "Dalc": 1,
        "Walc": 2,
        "health": 4,
        "absences": 4,
        "G1": 11,
        "G2": 11,
    },
    "At-Risk Student": {
        "school": "MS",
        "sex": "M",
        "age": 18,
        "address": "R",
        "famsize": "LE3",
        "Pstatus": "A",
        "Medu": 1,
        "Fedu": 1,
        "Mjob": "at_home",
        "Fjob": "other",
        "reason": "home",
        "guardian": "father",
        "traveltime": 3,
        "studytime": 1,
        "failures": 2,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "no",
        "higher": "no",
        "internet": "no",
        "romantic": "yes",
        "famrel": 2,
        "freetime": 4,
        "goout": 5,
        "Dalc": 3,
        "Walc": 4,
        "health": 3,
        "absences": 18,
        "G1": 7,
        "G2": 6,
    },
}


def load_combined_dataset() -> pd.DataFrame | None:
    """Load processed combined dataset read-only for exploratory overview charts."""
    path = PROCESSED_DIR / "student_combined_clean.csv"
    if path.is_file():
        return pd.read_csv(path)
    return None


def load_evaluation_data() -> dict[str, pd.DataFrame]:
    """Load saved Day 3 and Day 4 evaluation CSV tables from artifacts/."""
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in [
        ("ml", "ml_evaluation_results.csv"),
        ("dl", "dl_evaluation_results.csv"),
        ("comparison", "dl_vs_ml_comparison.csv"),
    ]:
        path = ARTIFACTS_DIR / filename
        if path.is_file():
            tables[key] = pd.read_csv(path)
    return tables


def load_unseen_test_rows(subject: str, limit: int = 25) -> tuple[pd.DataFrame, pd.Series] | None:
    """
    Load unseen test rows from data/raw/ matching the Day 3/Day 4 random_state=42 split.
    Used only to send real student records to POST /predict for live Actual vs Predicted plots.
    """
    filename = "student-mat.csv" if subject == "Mathematics" else "student-por.csv"
    path = RAW_DIR / filename
    if not path.is_file():
        return None

    df = pd.read_csv(path, sep=";")
    X = df.drop(columns=["G3"])
    y = df["G3"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test.iloc[:limit].copy(), y_test.iloc[:limit].copy()

