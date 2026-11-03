from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


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


def main() -> None:
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

    # ลบข้อมูลที่ใช้ไม่ได้และข้อมูลซ้ำ
    clean = df[[text_col, label_col]].dropna().copy()
    clean[text_col] = (
        clean[text_col]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    clean = clean[clean[text_col] != ""]
    clean = clean.drop_duplicates(subset=[text_col, label_col])

    X = clean[text_col]
    y = clean[label_col].astype(str)

    # แบ่งข้อความก่อนสร้าง Vocabulary ป้องกันข้อมูลจาก Test รั่วไป Train
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            (
                "bow",
                CountVectorizer(
                    lowercase=True,
                    min_df=2,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("กำลังฝึกโมเดล...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1_macro = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n=== ผลการประเมินโมเดล ===")
    print(f"จำนวน Train: {len(X_train):,}")
    print(f"จำนวน Test: {len(X_test):,}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Precision macro: {precision_macro:.4f}")
    print(f"Recall macro:    {recall_macro:.4f}")
    print(f"F1 macro:        {f1_macro:.4f}")
    print(f"F1 weighted:     {f1_weighted:.4f}")

    print("\n=== Classification Report ===")
    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
    print(report_text)

    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    metrics = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    with open(
        results_dir / "baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with open(
        results_dir / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report_text)

    labels = sorted(y.unique().tolist())
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=labels,
        display_labels=labels,
        xticks_rotation=45,
    )
    plt.title("Confusion Matrix: BoW + Logistic Regression")
    plt.tight_layout()
    plt.savefig(
        results_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    joblib.dump(
        pipeline,
        models_dir / "bow_logistic_regression.joblib",
    )

    print("\nบันทึกผลแล้ว:")
    print("- models/bow_logistic_regression.joblib")
    print("- results/baseline_metrics.json")
    print("- results/classification_report.txt")
    print("- results/confusion_matrix.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
