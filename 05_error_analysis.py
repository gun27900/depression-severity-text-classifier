from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_PATH = Path("models/best_text_classifier.joblib")


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


def export_top_features(model, output_path: Path, top_n: int = 20) -> None:
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["model"]

    if not hasattr(classifier, "coef_"):
        return

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(classifier.classes_):
        coefficients = classifier.coef_[class_index]
        top_indices = np.argsort(coefficients)[-top_n:][::-1]

        for rank, feature_index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_names[feature_index],
                    "coefficient": float(coefficients[feature_index]),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ไม่พบ models/best_text_classifier.joblib "
            "กรุณารัน py 04_compare_models.py ก่อน"
        )

    X, y = load_clean_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        confidence = probabilities.max(axis=1)
    else:
        confidence = np.full(len(X_test), np.nan)

    analysis = pd.DataFrame(
        {
            "text": X_test.reset_index(drop=True),
            "actual": y_test.reset_index(drop=True),
            "predicted": y_pred,
            "confidence": confidence,
        }
    )
    analysis["correct"] = analysis["actual"] == analysis["predicted"]

    errors = analysis[~analysis["correct"]].copy()
    errors = errors.sort_values("confidence", ascending=False)

    labels = sorted(y.unique().tolist())
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )

    confusion_pairs = (
        errors.groupby(["actual", "predicted"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    analysis.to_csv(
        results_dir / "all_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    errors.to_csv(
        results_dir / "misclassified_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    confusion_df.to_csv(
        results_dir / "confusion_table.csv",
        encoding="utf-8-sig",
    )
    confusion_pairs.to_csv(
        results_dir / "common_confusions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    export_top_features(
        model,
        results_dir / "top_features_by_class.csv",
        top_n=20,
    )

    print("=== การวิเคราะห์ข้อผิดพลาด ===")
    print(f"จำนวนข้อมูล Test: {len(analysis):,}")
    print(f"ทำนายถูก: {analysis['correct'].sum():,}")
    print(f"ทำนายผิด: {(~analysis['correct']).sum():,}")

    print("\n=== คู่คลาสที่สับสนกันบ่อยที่สุด ===")
    print(confusion_pairs.head(10).to_string(index=False))

    print("\nบันทึกผลแล้ว:")
    print("- results/all_test_predictions.csv")
    print("- results/misclassified_samples.csv")
    print("- results/confusion_table.csv")
    print("- results/common_confusions.csv")
    print("- results/top_features_by_class.csv")
    print(
        "\nข้อควรระวัง: ข้อความใน Dataset อาจมีเนื้อหาส่วนบุคคล "
        "ไม่ควรคัดลอกข้อความดิบลงรายงานโดยไม่ทำให้ไม่สามารถระบุตัวตนได้"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
