from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
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


def label_recall_scorer(label: str):
    return make_scorer(
        recall_score,
        labels=[label],
        average="macro",
        zero_division=0,
    )


def readable_weight(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ", ".join(
            f"{key}:{weight}" for key, weight in value.items()
        )
    return str(value)


def main() -> None:
    X, y = load_clean_data()

    pipeline = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    moderate_weights = {
        "minimal": 1.0,
        "mild": 1.5,
        "moderate": 1.25,
        "severe": 1.5,
    }

    stronger_weights = {
        "minimal": 1.0,
        "mild": 2.0,
        "moderate": 1.5,
        "severe": 2.0,
    }

    param_grid = {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [2, 3],
        "model__C": [0.5, 1.0, 2.0],
        "model__class_weight": [
            None,
            "balanced",
            moderate_weights,
            stronger_weights,
        ],
    }

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "recall_mild": label_recall_scorer("mild"),
        "recall_moderate": label_recall_scorer("moderate"),
        "recall_severe": label_recall_scorer("severe"),
    }

    total_configs = (
        len(param_grid["vectorizer__ngram_range"])
        * len(param_grid["vectorizer__min_df"])
        * len(param_grid["model__C"])
        * len(param_grid["model__class_weight"])
    )

    print("=== Logistic Regression Tuning ===")
    print(f"ข้อมูลหลังทำความสะอาด: {len(X):,} รายการ")
    print(f"จำนวนชุดพารามิเตอร์: {total_configs}")
    print(f"Cross-validation: {N_SPLITS} folds")
    print(
        "เกณฑ์เลือก: Mean Macro F1 จาก Dataset จริง "
        "(ไม่ใช้ Behavioral Test เลือกโมเดล)\n"
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(X, y)

    results = pd.DataFrame(search.cv_results_)

    output = pd.DataFrame(
        {
            "rank_f1_macro": results["rank_test_f1_macro"],
            "mean_f1_macro": results["mean_test_f1_macro"],
            "std_f1_macro": results["std_test_f1_macro"],
            "mean_accuracy": results["mean_test_accuracy"],
            "mean_f1_weighted": results["mean_test_f1_weighted"],
            "mean_recall_mild": results["mean_test_recall_mild"],
            "mean_recall_moderate": results["mean_test_recall_moderate"],
            "mean_recall_severe": results["mean_test_recall_severe"],
            "ngram_range": results["param_vectorizer__ngram_range"].astype(str),
            "min_df": results["param_vectorizer__min_df"],
            "C": results["param_model__C"],
            "class_weight": results["param_model__class_weight"].map(
                readable_weight
            ),
        }
    ).sort_values(
        by=["rank_f1_macro", "mean_recall_severe"],
        ascending=[True, False],
    )

    results_dir = Path("results")
    models_dir = Path("models")
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    output.to_csv(
        results_dir / "logistic_tuning_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top10 = output.head(10).copy()
    top10.to_csv(
        results_dir / "logistic_tuning_top10.csv",
        index=False,
        encoding="utf-8-sig",
    )

    best_index = int(search.best_index_)
    best_summary = {
        "selection_method": "5-fold cross-validation mean macro F1",
        "total_clean_records": int(len(X)),
        "tested_configurations": int(total_configs),
        "best_params": {
            key: (
                readable_weight(value)
                if key == "model__class_weight"
                else str(value)
            )
            for key, value in search.best_params_.items()
        },
        "best_cv_f1_macro_mean": float(
            results.loc[best_index, "mean_test_f1_macro"]
        ),
        "best_cv_f1_macro_std": float(
            results.loc[best_index, "std_test_f1_macro"]
        ),
        "best_cv_accuracy_mean": float(
            results.loc[best_index, "mean_test_accuracy"]
        ),
        "best_cv_f1_weighted_mean": float(
            results.loc[best_index, "mean_test_f1_weighted"]
        ),
        "best_cv_recall_mild_mean": float(
            results.loc[best_index, "mean_test_recall_mild"]
        ),
        "best_cv_recall_moderate_mean": float(
            results.loc[best_index, "mean_test_recall_moderate"]
        ),
        "best_cv_recall_severe_mean": float(
            results.loc[best_index, "mean_test_recall_severe"]
        ),
        "important_note": (
            "Behavioral Test เป็น Sanity Check เท่านั้น "
            "และไม่ได้ถูกใช้เลือกพารามิเตอร์"
        ),
    }

    with open(
        results_dir / "logistic_tuning_best_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best_summary, file, indent=2, ensure_ascii=False)

    # GridSearchCV refit โมเดลที่ดีที่สุดด้วยข้อมูลทั้งหมดแล้ว
    joblib.dump(
        search.best_estimator_,
        models_dir / "tuned_deployment_text_classifier.joblib",
    )

    print("\n=== 10 อันดับแรกตาม Mean Macro F1 ===")
    print(
        top10.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n=== โมเดลที่ดีที่สุด ===")
    print(f"Parameters: {search.best_params_}")
    print(
        f"Mean Macro F1: "
        f"{best_summary['best_cv_f1_macro_mean']:.4f} "
        f"(±{best_summary['best_cv_f1_macro_std']:.4f})"
    )
    print(
        f"Mean Accuracy: "
        f"{best_summary['best_cv_accuracy_mean']:.4f}"
    )
    print(
        f"Mean Mild Recall: "
        f"{best_summary['best_cv_recall_mild_mean']:.4f}"
    )
    print(
        f"Mean Moderate Recall: "
        f"{best_summary['best_cv_recall_moderate_mean']:.4f}"
    )
    print(
        f"Mean Severe Recall: "
        f"{best_summary['best_cv_recall_severe_mean']:.4f}"
    )

    print("\nบันทึกผลแล้ว:")
    print("- results/logistic_tuning_results.csv")
    print("- results/logistic_tuning_top10.csv")
    print("- results/logistic_tuning_best_summary.json")
    print("- models/tuned_deployment_text_classifier.joblib")

    print(
        "\nขั้นต่อไป: เปรียบเทียบโมเดลเดิมกับโมเดลที่ปรับแล้ว "
        "บน Behavioral Test ชุดเดิม โดยยังไม่เรียกผลนั้นว่า Accuracy"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
