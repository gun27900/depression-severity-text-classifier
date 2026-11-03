from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


MODEL_PATH = Path("models/deployment_text_classifier.joblib")
METADATA_PATH = Path("models/deployment_model_metadata.json")


def find_dataset() -> Path:
    candidates = [
        Path("data/Reddit_depression_dataset.csv"),
        Path("data/Reddit_depression_dataset (1).csv"),
        Path("Reddit_depression_dataset.csv"),
        Path("Reddit_depression_dataset (1).csv"),
    ]

    for path in candidates:
        if path.exists():
            return path

    csv_files = list(Path(".").glob("*.csv")) + list(Path("data").glob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "ไม่พบไฟล์ Dataset กรุณาใส่ไฟล์ CSV ไว้ในโฟลเดอร์ data/"
    )


def find_column(columns: list[str], choices: tuple[str, ...], kind: str) -> str:
    for name in choices:
        if name in columns:
            return name
    raise KeyError(f"ไม่พบคอลัมน์ {kind} คอลัมน์ที่พบคือ: {columns}")


def load_clean_data() -> tuple[pd.Series, pd.Series, Path]:
    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    text_col = find_column(
        df.columns.tolist(),
        ("text", "post", "content", "sentence"),
        "ข้อความ",
    )
    label_col = find_column(
        df.columns.tolist(),
        ("label", "class", "severity", "target"),
        "Label",
    )

    clean = df[[text_col, label_col]].dropna().copy()
    clean[text_col] = (
        clean[text_col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    clean[label_col] = (
        clean[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"minimum": "minimal"})
    )

    clean = clean[clean[text_col] != ""]
    clean = clean.drop_duplicates(subset=[text_col, label_col])

    return clean[text_col], clean[label_col], dataset_path


def build_selected_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    X, y, dataset_path = load_clean_data()
    model = build_selected_model()

    print("=== ฝึก Deployment Model ===")
    print("วิธี: TF-IDF Bigram + Balanced Logistic Regression")
    print(f"ข้อมูลหลังทำความสะอาด: {len(X):,} รายการ")
    print("\nจำนวนข้อมูลแต่ละคลาส:")
    print(y.value_counts().sort_index().to_string())

    print("\nกำลังฝึกโมเดลด้วยข้อมูลที่ทำความสะอาดทั้งหมด...")
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    vectorizer = model.named_steps["vectorizer"]

    metadata = {
        "model_purpose": "deployment/demo",
        "method": "TF-IDF Bigram + Balanced Logistic Regression",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_file": str(dataset_path),
        "training_records": int(len(X)),
        "label_distribution": {
            str(label): int(count)
            for label, count in y.value_counts().sort_index().items()
        },
        "classes": [str(label) for label in model.classes_],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "parameters": {
            "lowercase": True,
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "class_weight": "balanced",
            "max_iter": 3000,
            "random_state": 42,
        },
        "evaluation_note": (
            "โมเดลไฟล์นี้ฝึกด้วยข้อมูลที่ทำความสะอาดทั้งหมด "
            "จึงไม่ใช้คะแนนจากข้อมูลฝึกเป็นผลประเมิน "
            "ผลประเมินให้ใช้ไฟล์จาก 07_cross_validation_final_test.py"
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\nบันทึกสำเร็จ:")
    print(f"- {MODEL_PATH}")
    print(f"- {METADATA_PATH}")
    print(f"- Vocabulary size: {len(vectorizer.vocabulary_):,} features")
    print(
        "\nสำคัญ: โมเดลนี้สร้างไว้สำหรับทดลองกับข้อความใหม่ "
        "ค่าประเมินในรายงานต้องใช้ผลจาก 07_cross_validation_final_test.py"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
