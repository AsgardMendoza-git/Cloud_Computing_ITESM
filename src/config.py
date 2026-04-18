"""
config.py
---------
Project paths, Azure constants, and credential loading.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
UMBRAL_PATH = ARTIFACTS_DIR / "umbral.json"
SCORE_PATH = ARTIFACTS_DIR / "score.py"
URI_PATH = ARTIFACTS_DIR / "uri.json"

PREDICTION_CSV = DATA_DIR / "prediction_instances.csv"

SQL_TABLE = "SalesLT.Product"
AZURE_RESOURCE_GROUP = "Papus"
AZURE_WORKSPACE_NAME = "workspace"
AZURE_LOCATION = "eastus"
MODEL_NAME = "churn_model"
SERVICE_NAME = "churn-service-v2"


def load_credentials() -> dict:
    with open(CREDENTIALS_PATH, "r") as f:
        return json.load(f)


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
