"""
details.EAP/backend/run_space.py

Convenience entrypoint for a Hugging Face Space with SDK = Python.

Usage (Space command):
    python backend/run_space.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make the backend package importable when launched from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    # Uvicorn needs to find the app as backend.app:app from the repo root.
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
    ]
    print("Starting backend on port 7860 ...", flush=True)
    subprocess.run(cmd, check=True)
