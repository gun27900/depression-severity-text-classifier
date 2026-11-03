from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd


MODEL_PATH = Path("models/deployment_text_classifier.joblib")
RESULTS_DIR = Path("results")

DISPLAY_LABELS = {
    "minimal": "Minimal",
    "minimum": "Minimal",
    "mild": "Mild",
    "moderate": "Moderate",
    "severe": "Severe",
}

# เกณฑ์นี้ใช้เพื่อแจ้ง "ความไม่แน่นอนของโมเดล" เท่านั้น
TOP_SCORE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.10


BEHAVIORAL_TESTS = [
    # 1) ข้อความทั่วไป/ชีวิตประจำวัน
    {
        "test_id": "N01",
        "category": "neutral_daily",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "Today was a normal day. I went to class, finished my work, "
            "and talked with my friends."
        ),
    },
    {
        "test_id": "N02",
        "category": "neutral_daily",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I cooked dinner, watched a comedy, and had a relaxing evening."
        ),
    },
    {
        "test_id": "N03",
        "category": "neutral_daily",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I am looking forward to meeting my family this weekend."
        ),
    },
    {
        "test_id": "N04",
        "category": "neutral_daily",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "Work was busy today, but I completed everything I needed to do."
        ),
    },
    {
        "test_id": "N05",
        "category": "neutral_daily",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I feel happy and excited because I passed my exam."
        ),
    },

    # 2) อารมณ์เชิงลบเล็กน้อย แต่ยังทำกิจวัตรได้
    {
        "test_id": "M01",
        "category": "mild_negative",
        "heuristic_expected_tendency": "Mild",
        "text": (
            "I have been feeling a little sad lately, but I can still "
            "do my daily activities."
        ),
    },
    {
        "test_id": "M02",
        "category": "mild_negative",
        "heuristic_expected_tendency": "Mild",
        "text": (
            "I feel lonely sometimes, although I still enjoy spending time "
            "with my friends."
        ),
    },
    {
        "test_id": "M03",
        "category": "mild_negative",
        "heuristic_expected_tendency": "Mild",
        "text": (
            "I have less energy than usual, but I am still attending class "
            "and finishing my assignments."
        ),
    },
    {
        "test_id": "M04",
        "category": "mild_negative",
        "heuristic_expected_tendency": "Mild",
        "text": (
            "I have been worried and down for a few days, but I am managing."
        ),
    },
    {
        "test_id": "M05",
        "category": "mild_negative",
        "heuristic_expected_tendency": "Mild",
        "text": (
            "I do not feel like going out tonight, but I think I just need rest."
        ),
    },

    # 3) เริ่มกระทบชีวิตประจำวัน
    {
        "test_id": "O01",
        "category": "moderate_impact",
        "heuristic_expected_tendency": "Moderate",
        "text": (
            "For the past few weeks, I have lost interest in things I used "
            "to enjoy and it is becoming difficult to focus on my work."
        ),
    },
    {
        "test_id": "O02",
        "category": "moderate_impact",
        "heuristic_expected_tendency": "Moderate",
        "text": (
            "I have been avoiding people, missing classes, and struggling "
            "to complete basic tasks."
        ),
    },
    {
        "test_id": "O03",
        "category": "moderate_impact",
        "heuristic_expected_tendency": "Moderate",
        "text": (
            "I feel empty most days and I am finding it difficult to get "
            "motivated to do anything."
        ),
    },
    {
        "test_id": "O04",
        "category": "moderate_impact",
        "heuristic_expected_tendency": "Moderate",
        "text": (
            "My sleep has been poor for weeks and my mood is affecting "
            "my studies and relationships."
        ),
    },
    {
        "test_id": "O05",
        "category": "moderate_impact",
        "heuristic_expected_tendency": "Moderate",
        "text": (
            "I cannot concentrate, I have stopped enjoying my hobbies, "
            "and every day feels difficult."
        ),
    },

    # 4) ข้อความรุนแรง/ความเสี่ยงสูง
    # หมายเหตุ: ใช้เพื่อทดสอบโมเดลเท่านั้น ไม่ใช่ข้อความสำหรับวินิจฉัยบุคคล
    {
        "test_id": "S01",
        "category": "severe_risk_language",
        "heuristic_expected_tendency": "Severe",
        "text": (
            "I feel completely hopeless and I keep thinking that life "
            "is not worth living."
        ),
    },
    {
        "test_id": "S02",
        "category": "severe_risk_language",
        "heuristic_expected_tendency": "Severe",
        "text": (
            "I cannot get out of bed, I feel worthless, and I have thoughts "
            "of hurting myself."
        ),
    },
    {
        "test_id": "S03",
        "category": "severe_risk_language",
        "heuristic_expected_tendency": "Severe",
        "text": (
            "Everything feels unbearable and I do not see any reason to continue."
        ),
    },
    {
        "test_id": "S04",
        "category": "severe_risk_language",
        "heuristic_expected_tendency": "Severe",
        "text": (
            "I have stopped eating, stopped talking to people, and I feel "
            "that everyone would be better without me."
        ),
    },

    # 5) คำปฏิเสธและการดีขึ้น
    {
        "test_id": "R01",
        "category": "negation_recovery",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I am not depressed. I feel happy and excited about my life."
        ),
    },
    {
        "test_id": "R02",
        "category": "negation_recovery",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I used to feel hopeless, but now I feel much better and "
            "I am enjoying life again."
        ),
    },
    {
        "test_id": "R03",
        "category": "negation_recovery",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I was feeling down last month, but I have recovered and "
            "my mood is good now."
        ),
    },
    {
        "test_id": "R04",
        "category": "negation_recovery",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I am tired, but I am not sad or hopeless; I just need some sleep."
        ),
    },

    # 6) พูดถึงบุคคลอื่น/ตัวละคร ไม่ใช่ผู้เขียน
    {
        "test_id": "T01",
        "category": "third_person_context",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "My friend has been feeling depressed, but I am doing well "
            "and trying to support him."
        ),
    },
    {
        "test_id": "T02",
        "category": "third_person_context",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "The character in the movie felt hopeless, but I enjoyed the film."
        ),
    },
    {
        "test_id": "T03",
        "category": "third_person_context",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "My sister was very sad yesterday, while I felt calm and positive."
        ),
    },

    # 7) อาการจากสถานการณ์ทั่วไป
    {
        "test_id": "C01",
        "category": "situational_context",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I am exhausted because I studied all night for an important exam."
        ),
    },
    {
        "test_id": "C02",
        "category": "situational_context",
        "heuristic_expected_tendency": "Minimal",
        "text": (
            "I could not sleep because I was excited about starting my new job."
        ),
    },

    # 8) ข้อความสั้นและกำกวม
    {
        "test_id": "A01",
        "category": "short_ambiguous",
        "heuristic_expected_tendency": "Uncertain",
        "text": "I am tired.",
    },
    {
        "test_id": "A02",
        "category": "short_ambiguous",
        "heuristic_expected_tendency": "Uncertain",
        "text": "I am fine.",
    },
]


def normalize_label(label: str) -> str:
    return DISPLAY_LABELS.get(str(label).lower(), str(label))


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "ไม่พบ models/deployment_text_classifier.joblib "
            "กรุณารัน py 08_train_deployment_model.py ก่อน"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "โมเดลนี้ไม่รองรับ predict_proba จึงไม่สามารถทำ Behavioral Test นี้ได้"
        )

    class_names = [str(label).lower() for label in model.classes_]
    rows: list[dict[str, object]] = []

    print("=== Behavioral Test ===")
    print(f"จำนวนข้อความทดสอบ: {len(BEHAVIORAL_TESTS)}")
    print("หมายเหตุ: expected tendency เป็นเพียงสมมติฐานเพื่อ Sanity Check")
    print("ไม่ใช่ Ground Truth และห้ามใช้คำนวณ Accuracy ทางการแพทย์\n")

    for item in BEHAVIORAL_TESTS:
        text = str(item["text"])
        prediction = str(model.predict([text])[0]).lower()
        probabilities = model.predict_proba([text])[0]

        score_map = {
            class_name: float(score)
            for class_name, score in zip(class_names, probabilities)
        }

        ranked = sorted(
            score_map.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )

        top_label, top_score = ranked[0]
        second_label, second_score = ranked[1]
        margin = top_score - second_score

        low_top_score = top_score < TOP_SCORE_THRESHOLD
        small_margin = margin < MARGIN_THRESHOLD
        uncertain = low_top_score or small_margin

        expected = str(item["heuristic_expected_tendency"])
        heuristic_match: str

        if expected.lower() == "uncertain":
            heuristic_match = "not_scored"
        else:
            heuristic_match = (
                "yes"
                if normalize_label(prediction).lower() == expected.lower()
                else "no"
            )

        row = {
            "test_id": item["test_id"],
            "category": item["category"],
            "text": text,
            "heuristic_expected_tendency": expected,
            "prediction": normalize_label(prediction),
            "top_score": top_score,
            "second_prediction": normalize_label(second_label),
            "second_score": second_score,
            "top_two_margin": margin,
            "low_top_score": low_top_score,
            "small_margin": small_margin,
            "uncertain": uncertain,
            "heuristic_match": heuristic_match,
            "score_minimal": score_map.get(
                "minimal", score_map.get("minimum", 0.0)
            ),
            "score_mild": score_map.get("mild", 0.0),
            "score_moderate": score_map.get("moderate", 0.0),
            "score_severe": score_map.get("severe", 0.0),
            "review_notes": "",
        }
        rows.append(row)

        uncertainty_mark = "UNCERTAIN" if uncertain else "clearer"
        print(
            f"{item['test_id']} | {item['category']} | "
            f"{normalize_label(prediction)} "
            f"({top_score:.2%}) | margin={margin:.2%} | "
            f"{uncertainty_mark}"
        )

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)

    results.to_csv(
        RESULTS_DIR / "behavioral_test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_counts = (
        results["prediction"]
        .value_counts()
        .rename_axis("prediction")
        .reset_index(name="count")
    )
    prediction_counts["percentage"] = (
        prediction_counts["count"] / len(results) * 100
    ).round(2)

    prediction_counts.to_csv(
        RESULTS_DIR / "behavioral_test_prediction_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    category_summary = (
        results.groupby("category", as_index=False)
        .agg(
            test_count=("test_id", "count"),
            uncertain_count=("uncertain", "sum"),
            mean_top_score=("top_score", "mean"),
            mean_margin=("top_two_margin", "mean"),
        )
    )
    category_summary["uncertain_percentage"] = (
        category_summary["uncertain_count"]
        / category_summary["test_count"]
        * 100
    ).round(2)

    category_summary.to_csv(
        RESULTS_DIR / "behavioral_test_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    uncertain_samples = results[results["uncertain"]].copy()
    uncertain_samples.to_csv(
        RESULTS_DIR / "behavioral_test_uncertain_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    heuristic_scored = results[
        results["heuristic_match"].isin(["yes", "no"])
    ].copy()

    heuristic_agreement = (
        (heuristic_scored["heuristic_match"] == "yes").mean()
        if len(heuristic_scored) > 0
        else 0.0
    )

    print("\n=== สรุปคำทำนาย ===")
    print(prediction_counts.to_string(index=False))

    print("\n=== สรุปความไม่แน่นอน ===")
    uncertain_count = int(results["uncertain"].sum())
    print(
        f"ไม่แน่ชัด: {uncertain_count}/{len(results)} "
        f"({uncertain_count / len(results):.2%})"
    )
    print(
        f"คะแนนสูงสุดเฉลี่ย: {results['top_score'].mean():.2%}"
    )
    print(
        f"Top-two margin เฉลี่ย: {results['top_two_margin'].mean():.2%}"
    )

    print("\n=== Heuristic agreement สำหรับ Sanity Check เท่านั้น ===")
    print(
        f"ตรงกับแนวโน้มที่ตั้งไว้: "
        f"{heuristic_agreement:.2%} "
        f"จาก {len(heuristic_scored)} ข้อความที่นำมาเทียบ"
    )
    print(
        "ค่านี้ไม่ใช่ Accuracy เพราะข้อความถูกแต่งขึ้นและ "
        "ไม่ได้ผ่านการกำหนด Label จากผู้ประเมิน"
    )

    print("\n=== สรุปตามประเภทข้อความ ===")
    print(
        category_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nบันทึกผลแล้ว:")
    print("- results/behavioral_test_results.csv")
    print("- results/behavioral_test_prediction_counts.csv")
    print("- results/behavioral_test_category_summary.csv")
    print("- results/behavioral_test_uncertain_samples.csv")

    print(
        "\nการตีความ: Behavioral Test ใช้ตรวจพฤติกรรมและค้นหาจุดอ่อน "
        "ไม่ใช้แทนผลจาก Holdout Test หรือ Cross-validation"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
