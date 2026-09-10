# backend/

FastAPI backend for the Employee Attrition Prediction demo.

## Endpoints

- `POST /train` — run notebook-equivalent training pipeline and persist artifacts
- `GET /status` — whether artifacts are ready + whether a Groq key is configured
- `POST /predict` — employee input -> probability + SHAP factors
- `POST /narrative` — SHAP result -> HR narrative from Groq
- `GET /key-mode` — current Groq key mode (`custom` or `none`)
- `POST /set-key` — save a user-supplied Groq key (UI BYOK flow)
- `POST /clear-key` — remove the stored key

## Groq key (bring-your-own-key)

This project ships **no API key**. Users supply their own Groq key through the
UI `/set-key` flow; it is stored in `backend/.env.local` (git-ignored) and
never committed.

Optional server-side fallback: set the `GROQ_API_KEY` environment variable
(for example, as a Hugging Face Space secret). Without any key, prediction and
SHAP still work; only the HR narrative degrades gracefully.

## HF Space entrypoint

A lightweight runner is provided at `run_space.py`. On a Hugging Face Space
with SDK = Python, set the Space command to:

```bash
python backend/run_space.py
```

Or point the Space directly at `backend/app.py` if the Space supports a custom
app entrypoint.

## Artifacts

`POST /train` writes to `backend/artifacts/`:
- `model.joblib`
- `scaler.joblib`
- `explainer.joblib`
- `meta.json`

## Data

Place `employee_attrition_clean.csv` in `backend/data/` before training.
