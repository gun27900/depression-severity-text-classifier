from pathlib import Path
import sys

import joblib


MODEL_PATH = Path("models/bow_logistic_regression.joblib")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ไม่พบโมเดล กรุณารัน python 02_train_baseline.py ก่อน"
        )

    model = joblib.load(MODEL_PATH)

    print("พิมพ์ข้อความภาษาอังกฤษที่ต้องการทดลอง")
    print("พิมพ์ exit เพื่อออกจากโปรแกรม\n")

    while True:
        text = input("Text: ").strip()

        if text.lower() == "exit":
            break
        if not text:
            print("กรุณาพิมพ์ข้อความ\n")
            continue

        prediction = model.predict([text])[0]
        print(f"Prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]
            classes = model.classes_
            print("Probabilities:")
            for label, probability in sorted(
                zip(classes, probabilities),
                key=lambda item: item[1],
                reverse=True,
            ):
                print(f"  {label}: {probability:.2%}")

        print(
            "\nหมายเหตุ: ผลนี้เป็นการทดลองทางการศึกษา "
            "ไม่ใช่การวินิจฉัยทางการแพทย์\n"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
