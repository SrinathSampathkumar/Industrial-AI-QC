from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW = PROJECT_ROOT / "reports" / "raw_predictions.csv"
OUT = PROJECT_ROOT / "reports" / "benchmark_results.csv"

df = pd.read_csv(RAW)

results = []

print("="*70)
print("PATCHCORE BENCHMARK")
print("="*70)

for category in sorted(df["category"].unique()):

    cat = df[df["category"] == category]

    # Ground Truth
    y_true = (cat["true_label"] == "Anomaly").astype(int)

    # Prediction
    y_pred = (cat["prediction"] == "Anomaly").astype(int)

    # Score
    y_score = cat["score"]

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:
        auroc = roc_auc_score(
            y_true,
            y_score
        )
    except:
        auroc = 0

    threshold = cat["threshold"].mean()

    results.append({

        "category": category,

        "images": len(cat),

        "accuracy": round(accuracy,4),

        "precision": round(precision,4),

        "recall": round(recall,4),

        "f1": round(f1,4),

        "auroc": round(auroc,4),

        "threshold": round(threshold,4)

    })

result = pd.DataFrame(results)

result.to_csv(OUT,index=False)

print(result)

print("\n")
print("="*70)
print("TOP 3 BEST")
print("="*70)

print(result.sort_values(
    "f1",
    ascending=False
).head(3))

print("\n")
print("="*70)
print("TOP 3 WORST")
print("="*70)

print(result.sort_values(
    "f1",
    ascending=True
).head(3))

print("\nSaved to:")
print(OUT)