from __future__ import annotations

from pathlib import Path
import sys

import joblib


MODEL_PATH = Path("models/best_text_classifier.joblib")

DISPLAY_LABELS = {
    "minimum": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label).lower(), str(label))


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ไม่พบ models/best_text_classifier.joblib "
            "กรุณารัน py 04_compare_models.py ก่อน"
        )

    model = joblib.load(MODEL_PATH)

    print("=== Depression Severity Text Classifier ===")
    print("พิมพ์ข้อความภาษาอังกฤษเพื่อทดลอง")
    print("พิมพ์ exit เพื่อออก\n")

    while True:
        text = input("Text: ").strip()

        if text.lower() == "exit":
            break
        if not text:
            print("กรุณาพิมพ์ข้อความ\n")
            continue

        prediction = str(model.predict([text])[0])
        print(f"\nPrediction: {display_label(prediction)}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            ranked = sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )

            print("Probabilities:")
            for label, probability in ranked:
                print(f"  {display_label(str(label))}: {probability:.2%}")

        print(
            "\nหมายเหตุ: ผลนี้เป็นการทดลองทางการศึกษาและการคัดกรองเบื้องต้น "
            "ไม่ใช่การวินิจฉัยทางการแพทย์\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
