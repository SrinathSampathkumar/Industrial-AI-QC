"""
Generate Min-Max Scalers for PatchCore
======================================

Creates one scaler for each category so that
anomaly scores become comparable (0-100).

Author: Srinath
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "reports" / "raw_predictions.csv"

SCALER_DIR = PROJECT_ROOT / "models" / "scalers"

SCALER_DIR.mkdir(exist_ok=True)


df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("GENERATING PATCHCORE SCALERS")
print("=" * 70)

for category in sorted(df["category"].unique()):

    scores = (
        df[df["category"] == category]["score"]
        .values
        .reshape(-1, 1)
    )

    scaler = MinMaxScaler(feature_range=(0, 100))

    scaler.fit(scores)

    output = SCALER_DIR / f"{category}_scaler.pkl"

    joblib.dump(scaler, output)

    print(
        f"{category:12} "
        f"Min={scores.min():8.3f} "
        f"Max={scores.max():8.3f}"
    )

print("\nDone.")
print(f"Saved scalers to:\n{SCALER_DIR}")