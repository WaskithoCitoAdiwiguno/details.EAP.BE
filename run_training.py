"""
details.EAP/backend/run_training.py

Notebook-equivalent training pipeline for the demo.

Canonical live-demo model: Logistic Regression, matching the Gradio UI in the
notebook. This module:
  - loads employee_attrition_clean.csv
  - encodes categorical features (get_dummies)
  - scales numeric features (StandardScaler)
  - trains LogisticRegression
  - builds shap.LinearExplainer on the scaled training set
  - persists model, scaler, explainer, and metadata to artifacts_dir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CAT_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUM_COLS = [
    "Age",
    "DistanceFromHome",
    "Education",
    "JobLevel",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "NumCompaniesWorked",
    "MonthlyIncome",
    "PercentSalaryHike",
    "StockOptionLevel",
    "PerformanceRating",
    "TrainingTimesLastYear",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "JobInvolvement",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]


def _load_data(data_dir: Path) -> pd.DataFrame:
    csv_path = data_dir / "employee_attrition_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}. "
            "Place employee_attrition_clean.csv in backend/data/."
        )
    return pd.read_csv(csv_path)


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = df.copy()
    y = df["is_resign"]
    X = pd.get_dummies(df.drop(columns=["is_resign"]), columns=CAT_COLS)
    model_columns = list(X.columns)
    return X, y, model_columns


def _split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def _train_model(X_train_scaled: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model


def _evaluate_model(
    model: LogisticRegression,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    proba = model.predict_proba(X_test_scaled)[:, 1]
    pred = model.predict(X_test_scaled)
    report = classification_report(
        y_test,
        pred,
        target_names=["Stay", "Resign"],
        output_dict=True,
    )
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "recall_resign": float(report["Resign"]["recall"]),
        "f1_resign": float(report["Resign"]["f1-score"]),
    }


def _build_explainer(
    model: LogisticRegression,
    X_train_scaled: pd.DataFrame,
) -> Any:
    explainer = shap.LinearExplainer(model, X_train_scaled)
    return explainer


def train_and_save(
    *,
    force: bool = False,
    artifacts_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """Run the training pipeline and persist artifacts.

    Returns a summary dict with model name and key metrics.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_file = artifacts_dir / "model.joblib"
    scaler_file = artifacts_dir / "scaler.joblib"
    explainer_file = artifacts_dir / "explainer.joblib"
    meta_file = artifacts_dir / "meta.json"

    if not force and model_file.exists() and scaler_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return {
            "model_name": meta.get("model_name", "LogisticRegression"),
            "roc_auc": meta.get("roc_auc"),
            "recall_resign": meta.get("recall_resign"),
            "f1_resign": meta.get("f1_resign"),
            "message": "Artifacts already exist. Set force=True to retrain.",
        }

    df = _load_data(data_dir)
    X, y, model_columns = _prepare_xy(df)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = _split_and_scale(X, y)
    model = _train_model(X_train_scaled, y_train)
    metrics = _evaluate_model(model, X_test_scaled, y_test)
    explainer = _build_explainer(model, X_train_scaled)

    model_file.write_bytes(joblib.dumps(model))
    scaler_file.write_bytes(joblib.dumps(scaler))
    explainer_file.write_bytes(joblib.dumps(explainer))

    meta = {
        "model_name": "LogisticRegression",
        "model_columns": model_columns,
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "roc_auc": metrics["roc_auc"],
        "recall_resign": metrics["recall_resign"],
        "f1_resign": metrics["f1_resign"],
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "model_name": meta["model_name"],
        **metrics,
        "message": "Training complete and artifacts saved.",
    }
