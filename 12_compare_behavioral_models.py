from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd


OLD_MODEL_PATH = Path("models/deployment_text_classifier.joblib")
TUNED_MODEL_PATH = Path("models/tuned_deployment_text_classifier.joblib")
TEST_FILE = Path("results/behavioral_test_results.csv")
RESULTS_DIR = Path("results")

TOP_SCORE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10

DISPLAY_LABELS = {
    "minimal": "Minimal",
    "minimum": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label).lower(), str(label))


def evaluate_model(
    model,
    tests: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    classes = [str(label).lower() for label in model.classes_]

    for _, item in tests.iterrows():
        text = str(item["text"])
        expected = str(item["heuristic_expected_tendency"])

        prediction = str(model.predict([text])[0]).lower()
        probabilities = model.predict_proba([text])[0]

        score_map = {
            label: float(score)
            for label, score in zip(classes, probabilities)
        }

        ranked = sorted(
            score_map.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1]
        margin = top_score - second_score

        uncertain = (
            top_score < TOP_SCORE_THRESHOLD
            or margin < MARGIN_THRESHOLD
        )

        if expected.lower() == "uncertain":
            heuristic_match = "not_scored"
        else:
            heuristic_match = (
                "yes"
                if display_label(prediction).lower() == expected.lower()
                else "no"
            )

        rows.append(
            {
                "model": model_name,
                "test_id": item["test_id"],
                "category": item["category"],
                "text": text,
                "heuristic_expected_tendency": expected,
                "prediction": display_label(prediction),
                "top_score": top_score,
                "second_prediction": display_label(second_label),
                "second_score": second_score,
                "top_two_margin": margin,
                "uncertain": uncertain,
                "heuristic_match": heuristic_match,
                "score_minimal": score_map.get(
                    "minimal", score_map.get("minimum", 0.0)
                ),
                "score_mild": score_map.get("mild", 0.0),
                "score_moderate": score_map.get("moderate", 0.0),
                "score_severe": score_map.get("severe", 0.0),
            }
        )

    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []

    for model_name, group in results.groupby("model"):
        scored = group[group["heuristic_match"].isin(["yes", "no"])]
        agreement = (
            (scored["heuristic_match"] == "yes").mean()
            if len(scored) > 0
            else 0.0
        )

        prediction_counts = group["prediction"].value_counts()

        summaries.append(
            {
                "model": model_name,
                "test_count": len(group),
                "uncertain_count": int(group["uncertain"].sum()),
                "uncertain_percentage": float(group["uncertain"].mean()),
                "mean_top_score": float(group["top_score"].mean()),
                "mean_top_two_margin": float(
                    group["top_two_margin"].mean()
                ),
                "heuristic_agreement_sanity_only": float(agreement),
                "predicted_minimal": int(
                    prediction_counts.get("Minimal", 0)
                ),
                "predicted_mild": int(
                    prediction_counts.get("Mild", 0)
                ),
                "predicted_moderate": int(
                    prediction_counts.get("Moderate", 0)
                ),
                "predicted_severe": int(
                    prediction_counts.get("Severe", 0)
                ),
            }
        )

    return pd.DataFrame(summaries)


def main() -> None:
    missing = [
        str(path)
        for path in (OLD_MODEL_PATH, TUNED_MODEL_PATH, TEST_FILE)
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "ไม่พบไฟล์ที่จำเป็น:\n- " + "\n- ".join(missing)
        )

    tests = pd.read_csv(TEST_FILE)

    required_columns = {
        "test_id",
        "category",
        "text",
        "heuristic_expected_tendency",
    }
    missing_columns = required_columns - set(tests.columns)

    if missing_columns:
        raise KeyError(
            f"ไฟล์ Behavioral Test ขาดคอลัมน์: "
            f"{sorted(missing_columns)}"
        )

    old_model = joblib.load(OLD_MODEL_PATH)
    tuned_model = joblib.load(TUNED_MODEL_PATH)

    old_results = evaluate_model(
        old_model,
        tests,
        "Original deployment model",
    )
    tuned_results = evaluate_model(
        tuned_model,
        tests,
        "Tuned deployment model",
    )

    combined = pd.concat(
        [old_results, tuned_results],
        ignore_index=True,
    )

    summary = build_summary(combined)

    old_wide = old_results[
        [
            "test_id",
            "prediction",
            "top_score",
            "top_two_margin",
            "uncertain",
        ]
    ].rename(
        columns={
            "prediction": "old_prediction",
            "top_score": "old_top_score",
            "top_two_margin": "old_margin",
            "uncertain": "old_uncertain",
        }
    )

    tuned_wide = tuned_results[
        [
            "test_id",
            "prediction",
            "top_score",
            "top_two_margin",
            "uncertain",
        ]
    ].rename(
        columns={
            "prediction": "tuned_prediction",
            "top_score": "tuned_top_score",
            "top_two_margin": "tuned_margin",
            "uncertain": "tuned_uncertain",
        }
    )

    transitions = (
        tests[
            [
                "test_id",
                "category",
                "text",
                "heuristic_expected_tendency",
            ]
        ]
        .merge(old_wide, on="test_id")
        .merge(tuned_wide, on="test_id")
    )

    transitions["prediction_changed"] = (
        transitions["old_prediction"]
        != transitions["tuned_prediction"]
    )

    transitions["uncertainty_improved"] = (
        transitions["old_uncertain"]
        & ~transitions["tuned_uncertain"]
    )

    RESULTS_DIR.mkdir(exist_ok=True)

    combined.to_csv(
        RESULTS_DIR / "behavioral_models_detailed_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        RESULTS_DIR / "behavioral_models_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    transitions.to_csv(
        RESULTS_DIR / "behavioral_models_transitions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=== Behavioral Model Comparison ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    changed_count = int(transitions["prediction_changed"].sum())
    improved_count = int(transitions["uncertainty_improved"].sum())

    print("\n=== การเปลี่ยนแปลง ===")
    print(
        f"คำทำนายเปลี่ยน: {changed_count}/{len(transitions)} ข้อความ"
    )
    print(
        f"จากไม่แน่ชัดเป็นชัดขึ้น: "
        f"{improved_count}/{len(transitions)} ข้อความ"
    )

    print("\n=== รายการที่คำทำนายเปลี่ยน ===")
    changed = transitions[transitions["prediction_changed"]]

    if changed.empty:
        print("ไม่มีคำทำนายเปลี่ยน")
    else:
        print(
            changed[
                [
                    "test_id",
                    "category",
                    "old_prediction",
                    "tuned_prediction",
                    "old_top_score",
                    "tuned_top_score",
                ]
            ].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    print("\nบันทึกผลแล้ว:")
    print("- results/behavioral_models_detailed_comparison.csv")
    print("- results/behavioral_models_summary.csv")
    print("- results/behavioral_models_transitions.csv")

    print(
        "\nสำคัญ: ผลนี้ใช้เปรียบเทียบพฤติกรรมเท่านั้น "
        "ไม่ใช่ Accuracy และไม่ใช้แทน Cross-validation"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
