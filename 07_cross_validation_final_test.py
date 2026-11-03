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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
FINAL_TEST_SIZE = 0.20
N_SPLITS = 5


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
    clean[label_col] = (
        clean[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"minimum": "minimal"})
    )
    clean = clean[clean[text_col] != ""]
    clean = clean.drop_duplicates(subset=[text_col, label_col])

    return clean[text_col], clean[label_col]


def build_models() -> dict[str, Pipeline]:
    return {
        "Dummy majority baseline": Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(min_df=2)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "BoW + Logistic Regression": Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(lowercase=True, min_df=2)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "BoW + Balanced Logistic Regression": Pipeline(
            steps=[
                ("vectorizer", CountVectorizer(lowercase=True, min_df=2)),
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
        "BoW Bigram + Balanced Logistic Regression": Pipeline(
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
        "TF-IDF Bigram + Balanced Logistic Regression": Pipeline(
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


def calculate_final_metrics(y_true, y_pred) -> dict[str, float]:
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


def save_clean_distribution(y: pd.Series, results_dir: Path) -> None:
    counts = y.value_counts().sort_index()
    percentages = y.value_counts(normalize=True).sort_index().mul(100)

    summary = pd.DataFrame(
        {"count": counts, "percentage": percentages.round(2)}
    )

    summary.to_csv(
        results_dir / "clean_label_summary.csv",
        encoding="utf-8-sig",
    )

    ax = counts.plot(kind="bar")
    ax.set_title("Cleaned label distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        results_dir / "clean_label_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main() -> None:
    X, y = load_clean_data()

    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    save_clean_distribution(y, results_dir)

    # แยก Final Test Set ก่อน และห้ามใช้ส่วนนี้เลือกโมเดล
    X_dev, X_final_test, y_dev, y_final_test = train_test_split(
        X,
        y,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    models = build_models()
    cv_rows: list[dict[str, object]] = []

    print("=== โครงสร้างการประเมิน ===")
    print(f"ข้อมูลหลังทำความสะอาดทั้งหมด: {len(X):,}")
    print(f"Development Set สำหรับ Cross-validation: {len(X_dev):,}")
    print(f"Final Test Set ที่กันไว้: {len(X_final_test):,}")
    print(f"Cross-validation: {N_SPLITS} folds")
    print("\nกำลังเปรียบเทียบโมเดลด้วย Cross-validation...\n")

    for name, pipeline in models.items():
        print(f"กำลังประเมิน: {name}")

        scores = cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_precision_macro_mean": scores["test_precision_macro"].mean(),
            "cv_recall_macro_mean": scores["test_recall_macro"].mean(),
            "cv_f1_macro_mean": scores["test_f1_macro"].mean(),
            "cv_f1_macro_std": scores["test_f1_macro"].std(),
            "cv_f1_weighted_mean": scores["test_f1_weighted"].mean(),
        }
        cv_rows.append(row)

        print(
            f"  Mean Accuracy={row['cv_accuracy_mean']:.4f} | "
            f"Mean Macro F1={row['cv_f1_macro_mean']:.4f} "
            f"(±{row['cv_f1_macro_std']:.4f})"
        )

    cv_results = pd.DataFrame(cv_rows).sort_values(
        by=["cv_f1_macro_mean", "cv_accuracy_mean"],
        ascending=False,
    ).reset_index(drop=True)

    cv_results.to_csv(
        results_dir / "cross_validation_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== ผล Cross-validation ===")
    print(
        cv_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    best_name = str(cv_results.loc[0, "model"])
    best_model = models[best_name]

    print(f"\nโมเดลที่เลือกจาก Mean Macro F1: {best_name}")
    print("กำลังฝึกโมเดลนี้ด้วย Development Set ทั้งหมด...")

    best_model.fit(X_dev, y_dev)

    print("กำลังประเมิน Final Test Set ครั้งเดียว...")
    final_pred = best_model.predict(X_final_test)
    final_metrics = calculate_final_metrics(y_final_test, final_pred)

    print("\n=== Final Test Results ===")
    for metric_name, value in final_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    final_report = classification_report(
        y_final_test,
        final_pred,
        zero_division=0,
    )

    print("\n=== Final Classification Report ===")
    print(final_report)

    final_summary = {
        "selected_model": best_name,
        "selection_metric": "5-fold cross-validation mean macro F1",
        "total_clean_records": len(X),
        "development_size": len(X_dev),
        "final_test_size": len(X_final_test),
        **final_metrics,
    }

    with open(
        results_dir / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_summary, file, indent=2, ensure_ascii=False)

    with open(
        results_dir / "final_test_classification_report.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"Selected model: {best_name}\n")
        file.write("Selection method: 5-fold cross-validation mean macro F1\n\n")
        file.write(final_report)

    labels = sorted(y.unique().tolist())

    ConfusionMatrixDisplay.from_predictions(
        y_final_test,
        final_pred,
        labels=labels,
        display_labels=labels,
        xticks_rotation=45,
        normalize=None,
    )
    plt.title(f"Final Test Confusion Matrix\n{best_name}")
    plt.tight_layout()
    plt.savefig(
        results_dir / "final_test_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    ConfusionMatrixDisplay.from_predictions(
        y_final_test,
        final_pred,
        labels=labels,
        display_labels=labels,
        xticks_rotation=45,
        normalize="true",
        values_format=".2f",
    )
    plt.title(f"Normalized Final Test Confusion Matrix\n{best_name}")
    plt.tight_layout()
    plt.savefig(
        results_dir / "final_test_confusion_matrix_normalized.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    joblib.dump(
        best_model,
        models_dir / "final_evaluated_text_classifier.joblib",
    )

    print("\nบันทึกผลแล้ว:")
    print("- results/clean_label_summary.csv")
    print("- results/clean_label_distribution.png")
    print("- results/cross_validation_results.csv")
    print("- results/final_test_metrics.json")
    print("- results/final_test_classification_report.txt")
    print("- results/final_test_confusion_matrix.png")
    print("- results/final_test_confusion_matrix_normalized.png")
    print("- models/final_evaluated_text_classifier.joblib")

    print(
        "\nหมายเหตุ: ค่าจาก Final Test Set ชุดนี้ควรใช้เป็นผลหลักในรายงาน "
        "เพราะชุดนี้ไม่ได้ถูกใช้ในการเลือกโมเดล"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
