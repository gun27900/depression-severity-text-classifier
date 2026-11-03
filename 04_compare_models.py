from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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


RANDOM_STATE = 42
TEST_SIZE = 0.20


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


def load_clean_data() -> tuple[pd.Series, pd.Series]:
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
    clean[label_col] = clean[label_col].astype(str).str.strip().str.lower()

    clean = clean[clean[text_col] != ""]
    clean = clean.drop_duplicates(subset=[text_col, label_col])

    return clean[text_col], clean[label_col]


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }


def main() -> None:
    X, y = load_clean_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Dummy majority baseline": Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(min_df=2)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "BoW + Logistic Regression": Pipeline(
            steps=[
                (
                    "vectorizer",
                    CountVectorizer(
                        lowercase=True,
                        min_df=2,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "BoW + Balanced LR": Pipeline(
            steps=[
                (
                    "vectorizer",
                    CountVectorizer(
                        lowercase=True,
                        min_df=2,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "BoW bigram + Balanced LR": Pipeline(
            steps=[
                (
                    "vectorizer",
                    CountVectorizer(
                        lowercase=True,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "TF-IDF bigram + Balanced LR": Pipeline(
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
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }

    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    print("=== ข้อมูลที่ใช้ ===")
    print(f"Train: {len(X_train):,}")
    print(f"Test:  {len(X_test):,}")
    print("\nกำลังเปรียบเทียบโมเดล...\n")

    for name, pipeline in models.items():
        print(f"กำลังฝึก: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)

        results.append({"model": name, **metrics})
        fitted_models[name] = pipeline

        print(
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"Macro F1={metrics['f1_macro']:.4f}"
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["f1_macro", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    print("\n=== ตารางเปรียบเทียบ ===")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # เลือกโมเดลด้วย Macro F1 เพราะข้อมูลไม่สมดุล
    best_name = str(results_df.loc[0, "model"])
    best_model = fitted_models[best_name]
    best_pred = best_model.predict(X_test)

    print(f"\nโมเดลที่ดีที่สุดตาม Macro F1: {best_name}")
    print("\n=== Classification Report ของโมเดลที่ดีที่สุด ===")
    report = classification_report(
        y_test,
        best_pred,
        zero_division=0,
    )
    print(report)

    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    results_df.to_csv(
        results_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        results_dir / "best_model_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Best model: {best_name}\n\n")
        file.write(report)

    best_metrics = results_df.loc[0].to_dict()
    with open(
        results_dir / "best_model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_metrics, file, indent=2, ensure_ascii=False)

    labels = sorted(y.unique().tolist())
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_pred,
        labels=labels,
        display_labels=labels,
        xticks_rotation=45,
        normalize=None,
    )
    plt.title(f"Confusion Matrix\n{best_name}")
    plt.tight_layout()
    plt.savefig(
        results_dir / "best_model_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    joblib.dump(
        best_model,
        models_dir / "best_text_classifier.joblib",
    )

    print("\nบันทึกผลแล้ว:")
    print("- results/model_comparison.csv")
    print("- results/best_model_report.txt")
    print("- results/best_model_metrics.json")
    print("- results/best_model_confusion_matrix.png")
    print("- models/best_text_classifier.joblib")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
