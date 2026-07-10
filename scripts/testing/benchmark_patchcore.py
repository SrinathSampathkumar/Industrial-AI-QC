from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.inference.inference_patchcore import predict_image
from scripts.registry.model_registry import ModelRegistry
from pathlib import Path
import pandas as pd


DATASET_ROOT = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

rows = []

print("=" * 60)
print("RUNNING PATCHCORE BENCHMARK")
print("=" * 60)

for category in CATEGORIES:

    print(f"\nProcessing {category}")

    test_dir = DATASET_ROOT / category / "test"

    image_paths = sorted(test_dir.rglob("*.png"))

    for image_path in image_paths:

        true_label = "Normal" if image_path.parent.name == "good" else "Anomaly"

        result = predict_image(
            category=category,
            image_path=str(image_path)
        )

        rows.append({
            "category": category,
            "image": image_path.name,
            "defect_type": image_path.parent.name,
            "true_label": true_label,
            "prediction": result["prediction"],
            "score": result["anomaly_score"],
            "threshold": result["threshold"]
        })

print("\nSaving raw predictions...")

df = pd.DataFrame(rows)

csv_path = REPORTS_DIR / "raw_predictions.csv"

df.to_csv(csv_path, index=False)

print(f"Saved to: {csv_path}")
print(f"Total images processed: {len(df)}")