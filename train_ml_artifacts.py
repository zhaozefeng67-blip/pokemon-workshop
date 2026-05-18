"""Build ML artifact for the dashboard. Run from project root: python train_ml_artifacts.py"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

# Import training code without running Streamlit UI setup.
sys.modules["streamlit"] = MagicMock()
from dashboard.analysis.ml import _train_legendary_model_impl  # noqa: E402
from dashboard.config import ARTIFACTS_DIR, ML_ARTIFACT_PATH, ML_TEST_SIZE  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> None:
    df = pd.read_csv(PROJECT_DIR / "pokemon.csv")
    result = _train_legendary_model_impl(df, ML_TEST_SIZE)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ML_ARTIFACT_PATH, "wb") as artifact_file:
        pickle.dump(result, artifact_file)
    print(f"Saved: {ML_ARTIFACT_PATH}")
    print(result["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
