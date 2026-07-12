"""
Generate Benchmark Delta Report

Compares:
Week 1 Benchmark
vs
Current Benchmark

Output:
reports/benchmark_delta.csv
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEFORE = PROJECT_ROOT / "reports" / "benchmark_week1.csv"
AFTER = PROJECT_ROOT / "reports" / "benchmark_results.csv"

OUT = PROJECT_ROOT / "reports" / "benchmark_delta.csv"

if not BEFORE.exists():
    raise FileNotFoundError(
        f"Baseline benchmark not found:\n{BEFORE}"
    )

if not AFTER.exists():
    raise FileNotFoundError(
        f"Current benchmark not found:\n{AFTER}"
    )

before = pd.read_csv(BEFORE)
after = pd.read_csv(AFTER)

merged = before.merge(
    after,
    on="category",
    suffixes=("_before", "_after")
)

rows = []

for _, row in merged.iterrows():

    accuracy_before = row["accuracy_before"]
    accuracy_after = row["accuracy_after"]

    f1_before = row["f1_before"]
    f1_after = row["f1_after"]

    accuracy_delta = round(
        accuracy_after - accuracy_before,
        4
    )

    f1_delta = round(
        f1_after - f1_before,
        4
    )

    if accuracy_delta > 0.10:
        status = "Major Improvement"
    elif accuracy_delta > 0:
        status = "Improved"
    elif accuracy_delta == 0:
        status = "No Change"
    else:
        status = "Reduced"

    rows.append({

        # Category
        "category": row["category"],

        # Dataset
        "images": row["images_after"],

        # Accuracy
        "accuracy_before": accuracy_before,
        "accuracy_after": accuracy_after,
        "accuracy_delta": accuracy_delta,

        # F1
        "f1_before": f1_before,
        "f1_after": f1_after,
        "f1_delta": f1_delta,

        # Current Metrics
        "precision": row["precision_after"],
        "recall": row["recall_after"],
        "auroc": row["auroc_after"],
        "threshold": row["threshold_after"],

        # Training Configuration
        "backbone": "wide_resnet50_2",
        "layers": "layer2+layer3",
        "coreset_ratio": 0.10,
        "num_neighbors": 1,
        "batch_size": 8,

        # Experiment Details
        "experiment_name": "FineTune_v3",
        "experiment_date": "2026-07-11",

        # Model Version
        "model_version": "latest",

        # Final Status
        "status": status
    })

delta = pd.DataFrame(rows)

delta = delta.sort_values(
    by="accuracy_delta",
    ascending=False
)

delta.to_csv(
    OUT,
    index=False
)

print("=" * 70)
print("BENCHMARK DELTA")
print("=" * 70)

print(delta)

print("\nSaved to:")
print(OUT)