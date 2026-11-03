from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


DISPLAY_LABELS = {
    "minimal": "Minimal",
    "minimum": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}

THAI_LABELS = {
    "minimal": "ระดับต่ำ",
    "minimum": "ระดับต่ำ",
    "mild": "ระดับเล็กน้อย",
    "moderate": "ระดับปานกลาง",
    "severe": "ระดับรุนแรง",
}

TOP_SCORE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label).lower(), str(label))


def thai_label(label: str) -> str:
    return THAI_LABELS.get(str(label).lower(), str(label))


def load_model(model_path: str | Path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล: {path}. กรุณารัน 08_train_deployment_model.py ก่อน"
        )
    return joblib.load(path)


def analyze_text(model, text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("กรุณากรอกข้อความภาษาอังกฤษก่อนวิเคราะห์")

    predicted = str(model.predict([clean_text])[0]).lower()
    probabilities = model.predict_proba([clean_text])[0]

    ranked = sorted(
        [(str(label).lower(), float(score)) for label, score in zip(model.classes_, probabilities)],
        key=lambda item: item[1],
        reverse=True,
    )

    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1]
    margin = top_score - second_score

    uncertainty_reasons: list[str] = []
    if top_score < TOP_SCORE_THRESHOLD:
        uncertainty_reasons.append("คะแนนสูงสุดต่ำกว่า 50%")
    if margin < MARGIN_THRESHOLD:
        uncertainty_reasons.append("อันดับหนึ่งห่างจากอันดับสองน้อยกว่า 10%")

    scores = {
        display_label(label): score
        for label, score in ranked
    }

    return {
        "predicted": predicted,
        "top_label": top_label,
        "top_score": top_score,
        "second_label": second_label,
        "second_score": second_score,
        "margin": margin,
        "uncertain": bool(uncertainty_reasons),
        "uncertainty_reasons": uncertainty_reasons,
        "scores": scores,
    }


def explain_prediction(model, text: str, class_label: str, top_n: int = 8) -> dict[str, list[dict[str, float | str]]]:
    """Return local linear feature contributions for one Logistic Regression class.

    Contribution = TF-IDF value for the input x learned Logistic Regression coefficient.
    This explains the model's linear score, not a medical or causal relationship.
    """
    if not hasattr(model, "named_steps"):
        return {"supporting": [], "opposing": []}

    vectorizer = model.named_steps.get("vectorizer")
    classifier = model.named_steps.get("model")
    if vectorizer is None or classifier is None:
        return {"supporting": [], "opposing": []}

    if not hasattr(vectorizer, "get_feature_names_out") or not hasattr(classifier, "coef_"):
        return {"supporting": [], "opposing": []}

    class_names = [str(value).lower() for value in classifier.classes_]
    target = str(class_label).lower()
    if target not in class_names:
        return {"supporting": [], "opposing": []}

    class_index = class_names.index(target)
    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()

    row = vector.getrow(0)
    if row.nnz == 0:
        return {"supporting": [], "opposing": []}

    indices = row.indices
    tfidf_values = row.data
    coefficients = classifier.coef_[class_index, indices]
    contributions = tfidf_values * coefficients

    items = [
        {
            "feature": str(feature_names[idx]),
            "tfidf": float(tfidf),
            "coefficient": float(coef),
            "contribution": float(contribution),
        }
        for idx, tfidf, coef, contribution in zip(
            indices, tfidf_values, coefficients, contributions
        )
    ]

    supporting = sorted(
        (item for item in items if float(item["contribution"]) > 0),
        key=lambda item: float(item["contribution"]),
        reverse=True,
    )[:top_n]

    opposing = sorted(
        (item for item in items if float(item["contribution"]) < 0),
        key=lambda item: float(item["contribution"]),
    )[:top_n]

    return {"supporting": supporting, "opposing": opposing}
