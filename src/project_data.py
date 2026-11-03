from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"


def read_json(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_model_metadata() -> dict:
    path = MODELS_DIR / "deployment_model_metadata.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def project_metrics() -> dict[str, float | int | str]:
    final_metrics = read_json("final_test_metrics.json")
    metadata = read_model_metadata()
    cv = read_csv("cross_validation_results.csv")

    cv_macro_f1 = None
    cv_accuracy = None
    if not cv.empty:
        selected = cv.iloc[0]
        cv_macro_f1 = float(selected.get("cv_f1_macro_mean", 0.0))
        cv_accuracy = float(selected.get("cv_accuracy_mean", 0.0))

    return {
        "training_records": int(metadata.get("training_records", final_metrics.get("total_clean_records", 0) or 0)),
        "vocabulary_size": int(metadata.get("vocabulary_size", 0) or 0),
        "method": str(metadata.get("method", final_metrics.get("selected_model", ""))),
        "cv_macro_f1": cv_macro_f1 if cv_macro_f1 is not None else 0.0,
        "cv_accuracy": cv_accuracy if cv_accuracy is not None else 0.0,
        "holdout_accuracy": float(final_metrics.get("accuracy", 0.0) or 0.0),
        "holdout_macro_f1": float(final_metrics.get("f1_macro", 0.0) or 0.0),
        "holdout_precision_macro": float(final_metrics.get("precision_macro", 0.0) or 0.0),
        "holdout_recall_macro": float(final_metrics.get("recall_macro", 0.0) or 0.0),
    }
