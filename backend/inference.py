from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

MODEL_PATH = Path(__file__).resolve().parent / "models" / "deployment_text_classifier.joblib"
TOP_SCORE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10
CLEAR_SCORE_THRESHOLD = 0.65
CLEAR_MARGIN_THRESHOLD = 0.15

DISPLAY_LABELS = {
    "minimum": "Minimal",
    "minimal": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label).lower(), str(label).title())


def prediction_clarity(top_score: float, margin: float) -> str:
    if top_score < TOP_SCORE_THRESHOLD or margin < MARGIN_THRESHOLD:
        return "uncertain"
    if top_score >= CLEAR_SCORE_THRESHOLD and margin >= CLEAR_MARGIN_THRESHOLD:
        return "clearer"
    return "relatively_clear"


def explain_prediction(model, text: str, class_label: str, top_n: int = 7) -> dict[str, list[dict[str, Any]]]:
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
    row = vector.getrow(0)
    if row.nnz == 0:
        return {"supporting": [], "opposing": []}

    feature_names = vectorizer.get_feature_names_out()
    indices = row.indices
    tfidf_values = row.data
    coefficients = classifier.coef_[class_index, indices]
    contributions = tfidf_values * coefficients

    items = [
        {
            "feature": str(feature_names[idx]),
            "tfidf": round(float(tfidf), 6),
            "coefficient": round(float(coef), 6),
            "contribution": round(float(contribution), 6),
        }
        for idx, tfidf, coef, contribution in zip(
            indices, tfidf_values, coefficients, contributions
        )
    ]

    supporting = sorted(
        (item for item in items if item["contribution"] > 0),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]

    opposing = sorted(
        (item for item in items if item["contribution"] < 0),
        key=lambda item: item["contribution"],
    )[:top_n]

    return {"supporting": supporting, "opposing": opposing}


def analyze_text(text: str) -> dict[str, Any]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Text is required")

    model = load_model()
    probabilities = model.predict_proba([clean_text])[0]

    ranked = sorted(
        [
            {
                "label": str(label).lower(),
                "display_label": display_label(str(label)),
                "score": float(score),
            }
            for label, score in zip(model.classes_, probabilities)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    top = ranked[0]
    second = ranked[1]
    margin = float(top["score"] - second["score"])
    clarity = prediction_clarity(float(top["score"]), margin)

    explanation = explain_prediction(
        model=model,
        text=clean_text,
        class_label=str(top["label"]),
    )

    return {
        "top_label": top["label"],
        "top_display_label": top["display_label"],
        "top_score": top["score"],
        "second_label": second["label"],
        "second_display_label": second["display_label"],
        "second_score": second["score"],
        "margin": margin,
        "clarity": clarity,
        "uncertain": clarity == "uncertain",
        "scores": ranked,
        "explanation": explanation,
        "model": {
            "method": "TF-IDF (Unigram + Bigram) + Balanced Logistic Regression",
            "version": "portfolio-v3",
        },
    }
