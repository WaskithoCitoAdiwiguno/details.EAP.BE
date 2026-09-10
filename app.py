"""
details.EAP/backend/app.py

FastAPI backend for the Employee Attrition Prediction demo.

Bring-your-own-key (BYOK): no API key is bundled with this project.
Users supply their own Groq key through the UI (`/set-key`), which stores it
in `.env.local` (git-ignored). Optionally, the deployer may set the
`GROQ_API_KEY` environment variable as a server-side fallback.

Public endpoints:
  POST /train        - run notebook-equivalent training pipeline
  GET  /status       - artifact / model readiness status
  POST /predict      - employee inputs -> probability + SHAP explanation
  POST /narrative    - SHAP result -> HR narrative from Groq
  GET  /key-mode     - current Groq key mode ("custom" or "none")
  POST /set-key      - save a user-supplied Groq key
  POST /clear-key    - remove the stored key
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Groq key handling (bring-your-own-key)
#
# No key is bundled with this project. Priority:
#   1. Key supplied through the UI (/set-key), stored in .env.local
#   2. Environment variable GROQ_API_KEY (optional server-side fallback)
# If neither is set, /narrative degrades gracefully.
# ---------------------------------------------------------------------------
ENV_FALLBACK_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Custom key storage: kept out of version control via .gitignore
CUSTOM_KEY_FILE = PROJECT_ROOT / ".env.local"


def _read_custom_key() -> str:
    if not CUSTOM_KEY_FILE.exists():
        return ""
    try:
        return dotenv_values(str(CUSTOM_KEY_FILE)).get("GROQ_API_KEY", "")
    except Exception:
        return ""


def _write_custom_key(key: str) -> None:
    CUSTOM_KEY_FILE.write_text(f"GROQ_API_KEY={key}\n", encoding="utf-8")


def _clear_custom_key() -> None:
    if CUSTOM_KEY_FILE.exists():
        CUSTOM_KEY_FILE.unlink()


def active_groq_key() -> str:
    """Current effective Groq key: user-supplied if set, else env fallback."""
    custom = _read_custom_key()
    if custom:
        return custom
    return ENV_FALLBACK_GROQ_KEY


def has_groq_key() -> bool:
    """Whether any Groq key is currently configured."""
    return bool(active_groq_key())


KeyMode = Literal["custom", "none"]


def current_key_mode() -> KeyMode:
    if _read_custom_key():
        return "custom"
    if ENV_FALLBACK_GROQ_KEY:
        return "custom"  # env fallback still counts as "configured"
    return "none"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Employee Attrition Prediction backend",
    description="Training + inference backend for the EAP GitHub Pages demo.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    force: bool = Field(False, description="Re-run training even if artifacts exist.")


class TrainResponse(BaseModel):
    status: str
    model: str
    roc_auc: float | None = None
    recall_resign: float | None = None
    f1_resign: float | None = None
    message: str | None = None


class PredictRequest(BaseModel):
    employee: dict[str, Any] = Field(..., description="Raw employee feature dict.")


class SHAPFactor(BaseModel):
    feature: str
    contribution: float
    direction: str


class PredictResponse(BaseModel):
    probability: float
    label: str
    risk_tier: str
    top_factors: list[SHAPFactor]
    shap_values: dict[str, float]
    message: str | None = None


class NarrativeRequest(BaseModel):
    probability: float
    top_factors: list[SHAPFactor]
    employee_info: dict[str, Any] | None = None


class NarrativeResponse(BaseModel):
    narrative: str
    message: str | None = None


class KeyModeResponse(BaseModel):
    key_mode: KeyMode


class SetKeyRequest(BaseModel):
    groq_api_key: str = Field(..., min_length=1)


class SetKeyResponse(BaseModel):
    key_mode: KeyMode
    message: str


class ClearKeyResponse(BaseModel):
    key_mode: KeyMode
    message: str


class StatusResponse(BaseModel):
    ready: bool
    artifacts: dict[str, bool]
    key_mode: KeyMode
    key_available: bool


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


@app.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    model_file = ARTIFACTS / "model.joblib"
    scaler_file = ARTIFACTS / "scaler.joblib"
    explainer_file = ARTIFACTS / "explainer.joblib"
    meta_file = ARTIFACTS / "meta.json"

    ready = all(p.exists() for p in [model_file, scaler_file, explainer_file, meta_file])

    return StatusResponse(
        ready=ready,
        artifacts={
            "model": model_file.exists(),
            "scaler": scaler_file.exists(),
            "explainer": explainer_file.exists(),
            "meta": meta_file.exists(),
        },
        key_mode=current_key_mode(),
        key_available=has_groq_key(),
    )


# ---------------------------------------------------------------------------
# Train endpoint
# ---------------------------------------------------------------------------


@app.post("/train", response_model=TrainResponse)
def run_train(body: TrainRequest = TrainRequest()) -> TrainResponse:
    from .run_training import train_and_save

    try:
        result = train_and_save(force=body.force, artifacts_dir=ARTIFACTS, data_dir=DATA)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}") from exc

    return TrainResponse(
        status="ok",
        model=result["model_name"],
        roc_auc=result.get("roc_auc"),
        recall_resign=result.get("recall_resign"),
        f1_resign=result.get("f1_resign"),
        message=result.get("message"),
    )


# ---------------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if not (ARTIFACTS / "model.joblib").exists():
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")

    try:
        from .inference import predict_one
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction setup failed: {exc}") from exc

    try:
        result = predict_one(body.employee, artifacts_dir=ARTIFACTS)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        probability=result["probability"],
        label=result["label"],
        risk_tier=result["risk_tier"],
        top_factors=result["top_factors"],
        shap_values=result["shap_values"],
        message=result.get("message"),
    )


# ---------------------------------------------------------------------------
# Narrative endpoint
# ---------------------------------------------------------------------------


@app.post("/narrative", response_model=NarrativeResponse)
def narrative(body: NarrativeRequest) -> NarrativeResponse:
    try:
        from .narrative import generate_hr_narrative
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Narrative setup failed: {exc}") from exc

    try:
        text = generate_hr_narrative(
            body.probability,
            [t.model_dump() for t in body.top_factors],
            body.employee_info or {},
            api_key=active_groq_key(),
        )
    except Exception as exc:
        # Graceful degradation: story fails, but prediction + SHAP are still valid.
        return NarrativeResponse(
            narrative=f"[HR narrative unavailable: {type(exc).__name__}]",
            message="HR narrative call failed. Prediction and SHAP results are still valid.",
        )

    return NarrativeResponse(narrative=text)


# ---------------------------------------------------------------------------
# Groq key management endpoints
# ---------------------------------------------------------------------------


@app.get("/key-mode", response_model=KeyModeResponse)
def get_key_mode() -> KeyModeResponse:
    return KeyModeResponse(key_mode=current_key_mode())


@app.post("/set-key", response_model=SetKeyResponse)
def set_key(body: SetKeyRequest) -> SetKeyResponse:
    key = body.groq_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="groq_api_key cannot be empty.")
    _write_custom_key(key)
    return SetKeyResponse(
        key_mode="custom",
        message="Groq API key saved. The backend will use it until you clear it.",
    )


@app.post("/clear-key", response_model=ClearKeyResponse)
def clear_key() -> ClearKeyResponse:
    _clear_custom_key()
    if ENV_FALLBACK_GROQ_KEY:
        mode: KeyMode = "custom"
        msg = (
            "Groq API key removed. Falling back to the server-side "
            "GROQ_API_KEY environment variable."
        )
    else:
        mode = "none"
        msg = (
            "Groq API key removed. No key is configured anymore; "
            "the HR narrative will be unavailable until a new key is set."
        )
    return ClearKeyResponse(key_mode=mode, message=msg)
