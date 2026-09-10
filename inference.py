"""
details.EAP/backend/inference.py

Predict one employee from the persisted LR artifacts and compute SHAP factors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import shap


def _load_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    meta = json.loads((artifacts_dir / "meta.json").read_text(encoding="utf-8"))
    model = joblib.load(artifacts_dir / "model.joblib")
    scaler = joblib.load(artifacts_dir / "scaler.joblib")
    explainer = joblib.load(artifacts_dir / "explainer.joblib")
    return {
        "meta": meta,
        "model": model,
        "scaler": scaler,
        "explainer": explainer,
    }


def _build_row(raw: dict[str, Any], meta: dict[str, Any]) -> pd.DataFrame:
    num_cols = list(meta["num_cols"])
    cat_cols = list(meta["cat_cols"])
    model_columns = list(meta["model_columns"])

    row = {c: 0 for c in model_columns}

    for col in num_cols:
        row[col] = float(raw.get(col, 0))

    for col in cat_cols:
        value = raw.get(col)
        if value is None:
            continue
        key = f"{col}_{value}"
        if key in row:
            row[key] = 1

    return pd.DataFrame([row], columns=model_columns)


def predict_one(raw: dict[str, Any], artifacts_dir: Path) -> dict[str, Any]:
    artifacts = _load_artifacts(Path(artifacts_dir))
    meta = artifacts["meta"]
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    explainer = artifacts["explainer"]

    row = _build_row(raw, meta)
    row_scaled = scaler.transform(row)

    proba = float(model.predict_proba(row_scaled)[0, 1])
    label = "Resign" if proba >= 0.5 else "Stay"

    if proba < 0.30:
        risk_tier = "RENDAH"
    elif proba < 0.60:
        risk_tier = "SEDANG"
    else:
        risk_tier = "TINGGI"

    shap_exp = explainer(row_scaled)
    shap_series = pd.Series(shap_exp.values[0], index=list(meta["model_columns"]))
    shap_series = shap_series.sort_values(key=lambda s: s.abs(), ascending=False)

    top = shap_series.head(5)
    top_factors = [
        {
            "feature": str(f),
            "contribution": float(v),
            "direction": "meningkatkan risiko resign" if v > 0 else "menurunkan risiko resign",
        }
        for f, v in top.items()
    ]

    shap_values = {str(f): float(v) for f, v in shap_series.items()}

    return {
        "probability": proba,
        "label": label,
        "risk_tier": risk_tier,
        "top_factors": top_factors,
        "shap_values": shap_values,
    }
