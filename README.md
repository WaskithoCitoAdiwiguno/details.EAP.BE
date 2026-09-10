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

## HF Space deployment (Docker SDK)

The repo ships a `Dockerfile`, so it runs directly as a Hugging Face Space:

1. Create a Space (SDK = **Docker**, app port 7860).
2. Paste the contents of `README.md.header` at the top of the Space's README
   (this sets `sdk: docker` and `app_port: 7860`).
3. Add the optional `GROQ_API_KEY` secret under Settings → Variables and secrets.
4. Upload/push these files to the Space. Uvicorn serves `app:app` on port 7860.

The legacy `run_space.py` runner (SDK = Python, package layout) is kept for
reference but is no longer the recommended path.

## Artifacts

`POST /train` writes to `backend/artifacts/`:
- `model.joblib`
- `scaler.joblib`
- `explainer.joblib`
- `meta.json`

## Data

Place `employee_attrition_clean.csv` in `backend/data/` before training.
