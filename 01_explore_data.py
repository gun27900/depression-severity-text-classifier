from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


def find_dataset() -> Path:
    """ค้นหาไฟล์ Dataset จากตำแหน่งที่พบบ่อย"""
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
        "ไม่พบไฟล์ CSV กรุณานำ Dataset ไปไว้ในโฟลเดอร์ data/ "
        "และตั้งชื่อว่า Reddit_depression_dataset.csv"
    )


def find_column(columns: list[str], choices: tuple[str, ...], kind: str) -> str:
    for name in choices:
        if name in columns:
            return name
    raise KeyError(
        f"ไม่พบคอลัมน์ {kind} คอลัมน์ที่พบคือ: {columns}"
    )


def main() -> None:
    dataset_path = find_dataset()
    print(f"กำลังอ่านไฟล์: {dataset_path.resolve()}")

    df = pd.read_csv(dataset_path)

    # ทำชื่อคอลัมน์ให้เป็นมาตรฐาน เช่น Text -> text
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

    print("\n=== โครงสร้างข้อมูล ===")
    print(f"จำนวนข้อมูลทั้งหมด: {len(df):,} แถว")
    print(f"ชื่อคอลัมน์: {df.columns.tolist()}")
    print(f"คอลัมน์ข้อความ: {text_col}")
    print(f"คอลัมน์คำตอบ: {label_col}")

    print("\n=== ตัวอย่างข้อมูล 5 แถวแรก ===")
    preview = df[[text_col, label_col]].head().copy()
    preview[text_col] = (
        preview[text_col]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 120)
    )
    print(preview.to_string(index=True))

    missing_text = int(df[text_col].isna().sum())
    missing_label = int(df[label_col].isna().sum())
    duplicates = int(df.duplicated(subset=[text_col, label_col]).sum())

    print("\n=== ตรวจคุณภาพข้อมูล ===")
    print(f"ข้อความว่าง: {missing_text:,} แถว")
    print(f"Label ว่าง: {missing_label:,} แถว")
    print(f"ข้อมูลซ้ำ (text + label): {duplicates:,} แถว")

    valid_text = df[text_col].fillna("").astype(str).str.strip()
    lengths = valid_text.str.split().str.len()

    print("\n=== ความยาวข้อความ (จำนวนคำโดยประมาณ) ===")
    print(f"ค่าเฉลี่ย: {lengths.mean():.2f} คำ")
    print(f"ค่ากลาง: {lengths.median():.2f} คำ")
    print(f"สั้นที่สุด: {lengths.min():.0f} คำ")
    print(f"ยาวที่สุด: {lengths.max():.0f} คำ")

    counts = df[label_col].value_counts(dropna=False)
    percentages = df[label_col].value_counts(
        normalize=True, dropna=False
    ).mul(100)

    summary = pd.DataFrame(
        {"count": counts, "percentage": percentages.round(2)}
    )

    print("\n=== จำนวนข้อมูลแต่ละ Label ===")
    print(summary.to_string())

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    summary.to_csv(
        results_dir / "label_summary.csv",
        encoding="utf-8-sig",
    )

    ax = counts.plot(kind="bar")
    ax.set_title("Label distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        results_dir / "label_distribution.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

    print("\nบันทึกผลแล้ว:")
    print("- results/label_summary.csv")
    print("- results/label_distribution.png")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nเกิดข้อผิดพลาด: {exc}", file=sys.stderr)
        sys.exit(1)
