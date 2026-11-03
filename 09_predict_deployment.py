from __future__ import annotations

from pathlib import Path
import sys

import joblib


MODEL_PATH = Path("models/deployment_text_classifier.joblib")

DISPLAY_LABELS = {
    "minimal": "Minimal",
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
            "ไม่พบ models/deployment_text_classifier.joblib "
            "กรุณารัน py 08_train_deployment_model.py ก่อน"
        )

    model = joblib.load(MODEL_PATH)

    print("=== Deployment Text Classifier ===")
    print("รองรับข้อความภาษาอังกฤษในต้นแบบ")
    print("พิมพ์ exit เพื่อออก\n")

    while True:
        text = input("Text: ").strip()

        if text.lower() == "exit":
            print("ปิดโปรแกรมแล้ว")
            break

        if not text:
            print("กรุณาพิมพ์ข้อความ\n")
            continue

        predicted = str(model.predict([text])[0])
        print(f"\nPrediction: {display_label(predicted)}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            ranked = sorted(
                zip(model.classes_, probabilities),
                key=lambda item: item[1],
                reverse=True,
            )

            print("Model scores:")
            for label, probability in ranked:
                print(f"  {display_label(str(label))}: {probability:.2%}")

            top_probability = float(ranked[0][1])
            second_probability = float(ranked[1][1])
            margin = top_probability - second_probability

            print(f"Top-two margin: {margin:.2%}")

            if top_probability < 0.50 or margin < 0.10:
                print(
                    "คำเตือน: คะแนนสูงสุดต่ำหรือใกล้กับอันดับสอง "
                    "ผลนี้จึงมีความไม่แน่นอนค่อนข้างสูง"
                )

        print(
            "\nหมายเหตุ: คะแนนเป็นผลการจำแนกระหว่างคลาสของโมเดล "
            "ไม่ใช่เปอร์เซ็นต์ความเสี่ยงทางการแพทย์ "
            "และไม่ใช่การวินิจฉัยโรค\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
