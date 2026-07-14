"""Compare the Week-2 calibrated benchmark with the immutable Week-1 baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_WEEK1 = REPORTS_DIR / "benchmark_week1.csv"
DEFAULT_WEEK2 = REPORTS_DIR / "benchmark_results.csv"
DEFAULT_OUTPUT = REPORTS_DIR / "benchmark_delta.csv"
REQUIRED_COLUMNS = (
    "category", "week1_accuracy", "week2_accuracy", "week1_f1", "week2_f1",
    "accuracy_improvement", "f1_improvement", "threshold_used", "normalized_score_range",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the calibrated Week-2 benchmark delta report.")
    parser.add_argument("--week1", type=Path, default=DEFAULT_WEEK1)
    parser.add_argument("--week2", type=Path, default=DEFAULT_WEEK2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def generate_delta(week1_path: Path, week2_path: Path) -> pd.DataFrame:
    if not week1_path.is_file() or not week2_path.is_file():
        raise FileNotFoundError("Both Week-1 and Week-2 benchmark CSV files are required.")
    week1 = pd.read_csv(week1_path)
    week2 = pd.read_csv(week2_path)
    merged = week1[["category", "accuracy", "f1"]].merge(
        week2[["category", "accuracy", "f1", "threshold_used", "normalized_score_range"]],
        on="category", validate="one_to_one", suffixes=("_week1", "_week2"),
    )
    delta = pd.DataFrame({
        "category": merged["category"],
        "week1_accuracy": merged["accuracy_week1"].round(4),
        "week2_accuracy": merged["accuracy_week2"].round(4),
        "week1_f1": merged["f1_week1"].round(4),
        "week2_f1": merged["f1_week2"].round(4),
        "accuracy_improvement": (merged["accuracy_week2"] - merged["accuracy_week1"]).round(4),
        "f1_improvement": (merged["f1_week2"] - merged["f1_week1"]).round(4),
        "threshold_used": merged["threshold_used"].round(4),
        "normalized_score_range": merged["normalized_score_range"],
    })
    if tuple(delta.columns) != REQUIRED_COLUMNS:
        raise RuntimeError("Delta report schema is not the required Week-2 schema.")
    return delta.sort_values("category").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    delta = generate_delta(args.week1, args.week2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    delta.to_csv(args.output, index=False)

    best = delta.loc[delta["accuracy_improvement"].idxmax()]
    worst = delta.loc[delta["accuracy_improvement"].idxmin()]
    print("WEEK-2 BENCHMARK SUMMARY")
    print(f"Average accuracy improvement: {delta['accuracy_improvement'].mean():+.4f}")
    print(f"Average F1 improvement: {delta['f1_improvement'].mean():+.4f}")
    print(f"Best category: {best['category']} ({best['accuracy_improvement']:+.4f} accuracy)")
    print(f"Worst category: {worst['category']} ({worst['accuracy_improvement']:+.4f} accuracy)")
    print(f"Execution time: {time.perf_counter() - started:.2f}s")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
