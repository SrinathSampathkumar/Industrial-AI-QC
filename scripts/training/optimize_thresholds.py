"""
Adaptive Threshold Optimizer
============================

Optimizes PatchCore anomaly thresholds for every
MVTec category by maximizing F1 score.

Author: Srinath
"""

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW = PROJECT_ROOT / "reports" / "raw_predictions.csv"

OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "thresholds"
    / "thresholds.json"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)

thresholds = {}

print("=" * 70)
print("PATCHCORE ADAPTIVE THRESHOLD OPTIMIZER")
print("=" * 70)

for category in sorted(df["category"].unique()):

    cat = df[df["category"] == category]

    y_true = (
        cat["true_label"] == "Anomaly"
    ).astype(int).values

    scores = cat["score"].values

    best_threshold = None
    best_f1 = -1

    candidate_thresholds = np.linspace(
        scores.min(),
        scores.max(),
        200
    )

    for threshold in candidate_thresholds:

        y_pred = (
            scores >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    thresholds[category] = {

        "threshold": round(float(best_threshold), 4),

        "best_f1": round(float(best_f1), 4),

        "min_score": round(float(scores.min()), 4),

        "max_score": round(float(scores.max()), 4),

        "validation_images": int(len(scores)),

        "optimized_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    }

    print(
        f"{category:<12}"
        f" Threshold={best_threshold:.4f}"
        f"  F1={best_f1:.4f}"
        f"  Images={len(scores)}"
    )

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        thresholds,
        f,
        indent=4
    )

print("\n" + "=" * 70)
print("Optimization Complete")
print("=" * 70)
print(f"Saved to:\n{OUTPUT}")