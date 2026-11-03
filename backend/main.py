from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import analyze_text, load_model

app = FastAPI(
    title="Depression Severity Classifier API",
    version="3.0.0",
    description=(
        "Educational NLP inference API using TF-IDF and Balanced Logistic Regression. "
        "Outputs are model scores, not medical diagnoses."
    ),
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/api")
def api_root():
    return {
        "name": "Depression Severity Classifier API",
        "version": "3.0.0",
        "status": "ok",
    }


@app.get("/api/health")
def health():
    try:
        model = load_model()
        classes = [str(value) for value in model.classes_]
        return {"status": "ok", "model_loaded": True, "classes": classes}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model load failed: {exc}") from exc


@app.get("/api/metadata")
def metadata():
    return {
        "method": "TF-IDF (Unigram + Bigram) + Balanced Logistic Regression",
        "training_records": 3534,
        "vocabulary_size": 36151,
        "classes": ["Minimal", "Mild", "Moderate", "Severe"],
        "cv_macro_f1": 0.4424415559956588,
        "holdout_accuracy": 0.7057991513437057,
        "holdout_macro_f1": 0.45604848626272765,
        "purpose": "Educational / portfolio prototype",
    }


@app.post("/api/predict")
def predict(payload: PredictRequest):
    try:
        return analyze_text(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
