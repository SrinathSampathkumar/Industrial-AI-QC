"""Create the calibrated Week-2 PatchCore benchmark.

The score snapshot in ``reports/raw_predictions.csv`` contains the model's raw
PatchCore scores for every MVTec test image.  This script intentionally applies
the current registry thresholds at benchmark time instead of trusting a label
that may have been produced with a previous threshold configuration.

Use ``--verify-model-loads`` for a checkpoint-load health check before creating
the report.  It loads one category at a time and releases it afterwards.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.registry.model_registry import ModelRegistry


CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)
REPORTS_DIR = PROJECT_ROOT / "reports"
CALIBRATION_FILE = PROJECT_ROOT / "models" / "thresholds" / "thresholds.json"
DEFAULT_RAW_INPUT = REPORTS_DIR / "raw_predictions.csv"
DEFAULT_WEEK2_RAW_OUTPUT = REPORTS_DIR / "raw_predictions_week2.csv"
DEFAULT_RESULTS_OUTPUT = REPORTS_DIR / "benchmark_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the calibrated Week-2 PatchCore benchmark.")
    parser.add_argument("--raw-input", type=Path, default=DEFAULT_RAW_INPUT,
                        help="CSV containing category, true_label and raw score columns")
    parser.add_argument("--week2-raw-output", type=Path, default=DEFAULT_WEEK2_RAW_OUTPUT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--verify-model-loads", action="store_true",
                        help="Load every category checkpoint before evaluating the score snapshot")
    return parser.parse_args()


def load_calibration() -> dict[str, dict]:
    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(f"Calibrated thresholds not found: {CALIBRATION_FILE}")
    with CALIBRATION_FILE.open(encoding="utf-8") as file:
        calibration = json.load(file)
    missing = sorted(set(CATEGORIES) - set(calibration))
    if missing:
        raise ValueError(f"Calibrated thresholds are missing categories: {', '.join(missing)}")
    return calibration


def verify_registry(registry: ModelRegistry, calibration: dict[str, dict], load_models: bool) -> None:
    """Ensure every category is trained and registry threshold resolution is calibrated."""
    failures = []
    for category in CATEGORIES:
        if not registry.is_trained(category):
            failures.append(f"{category}: checkpoint unavailable")
            continue
        threshold = registry.get_threshold(category)
        expected = float(calibration[category]["threshold"])
        if threshold != expected:
            failures.append(f"{category}: registry={threshold} calibration={expected}")
            continue
        if load_models:
            try:
                registry.load(category)
            except Exception as error:
                failures.append(f"{category}: {error}")
            finally:
                # This is only a health check. Do not retain 15 large models in memory.
                registry._models.pop(category, None)
                gc.collect()
    if failures:
        raise RuntimeError("Registry verification failed:\n- " + "\n- ".join(failures))


def read_raw_predictions(raw_input: Path) -> pd.DataFrame:
    if not raw_input.is_file():
        raise FileNotFoundError(f"Raw prediction snapshot not found: {raw_input}")
    raw = pd.read_csv(raw_input)
    required_columns = {"category", "true_label", "score"}
    missing = sorted(required_columns - set(raw.columns))
    if missing:
        raise ValueError(f"Raw prediction CSV is missing columns: {', '.join(missing)}")
    found_categories = set(raw["category"].dropna().unique())
    missing_categories = sorted(set(CATEGORIES) - found_categories)
    unexpected_categories = sorted(found_categories - set(CATEGORIES))
    if missing_categories or unexpected_categories:
        raise ValueError(
            f"Raw prediction categories do not match MVTec-15; "
            f"missing={missing_categories}, unexpected={unexpected_categories}"
        )
    raw["score"] = pd.to_numeric(raw["score"], errors="raise")
    return raw


def evaluate_calibrated_scores(raw: pd.DataFrame, registry: ModelRegistry,
                               calibration: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply calibrated thresholds and calculate one Week-2 metric row per category."""
    calibrated_rows = []
    metric_rows = []

    for category in CATEGORIES:
        category_rows = raw[raw["category"] == category].copy()
        threshold = registry.get_threshold(category)
        score_min = float(calibration[category]["min_score"])
        score_max = float(calibration[category]["max_score"])
        denominator = score_max - score_min
        if denominator <= 0:
            normalized_scores = pd.Series(0.0, index=category_rows.index)
        else:
            normalized_scores = ((category_rows["score"] - score_min) / denominator * 100).clip(0, 100)

        category_rows["threshold"] = threshold
        category_rows["prediction"] = category_rows["score"].ge(threshold).map(
            {True: "Anomaly", False: "Normal"}
        )
        category_rows["normalized_score"] = normalized_scores
        calibrated_rows.append(category_rows)

        true_values = category_rows["true_label"].eq("Anomaly").astype(int)
        predicted_values = category_rows["prediction"].eq("Anomaly").astype(int)
        try:
            auroc = roc_auc_score(true_values, category_rows["score"])
        except ValueError:
            auroc = 0.0
        metric_rows.append({
            "category": category,
            "images": int(len(category_rows)),
            "accuracy": round(float(accuracy_score(true_values, predicted_values)), 4),
            "precision": round(float(precision_score(true_values, predicted_values, zero_division=0)), 4),
            "recall": round(float(recall_score(true_values, predicted_values, zero_division=0)), 4),
            "f1": round(float(f1_score(true_values, predicted_values, zero_division=0)), 4),
            "auroc": round(float(auroc), 4),
            "threshold_used": round(float(threshold), 4),
            "normalized_score_range": (
                f"{normalized_scores.min():.2f}-{normalized_scores.max():.2f}"
            ),
        })

    return pd.concat(calibrated_rows, ignore_index=True), pd.DataFrame(metric_rows)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration()
    registry = ModelRegistry()
    print("Verifying calibrated registry thresholds for all 15 categories...")
    verify_registry(registry, calibration, args.verify_model_loads)
    print("Registry verification passed.")

    raw = read_raw_predictions(args.raw_input)
    week2_raw, results = evaluate_calibrated_scores(raw, registry, calibration)
    args.week2_raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    week2_raw.to_csv(args.week2_raw_output, index=False)
    results.to_csv(args.results_output, index=False)

    print("\nWEEK-2 CALIBRATED PATCHCORE BENCHMARK")
    print(results.to_string(index=False))
    print(f"\nRaw predictions: {args.week2_raw_output}")
    print(f"Metrics: {args.results_output}")
    print(f"Execution time: {time.perf_counter() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
