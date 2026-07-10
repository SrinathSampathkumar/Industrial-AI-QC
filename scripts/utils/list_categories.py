from pathlib import Path
import csv

# Root dataset folder
DATASET_ROOT = Path("datasets")

# Output CSV
REPORT_DIR = Path("reports/dataset")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = REPORT_DIR / "categories_summary.csv"

rows = []

print("=" * 70)
print("Scanning MVTec Dataset")
print("=" * 70)

for category in sorted(DATASET_ROOT.iterdir()):

    if not category.is_dir():
        continue

    train_good = category / "train" / "good"
    test_dir = category / "test"
    gt_dir = category / "ground_truth"

    train_images = len(list(train_good.glob("*.*"))) if train_good.exists() else 0

    test_images = 0
    defect_types = []

    if test_dir.exists():
        for folder in test_dir.iterdir():

            if folder.is_dir():

                defect_types.append(folder.name)

                test_images += len(list(folder.glob("*.*")))

    rows.append([
        category.name,
        train_images,
        test_images,
        len(defect_types),
        train_good.exists(),
        test_dir.exists(),
        gt_dir.exists()
    ])

    print(f"{category.name:12} "
          f"Train={train_images:4} "
          f"Test={test_images:4} "
          f"Defects={len(defect_types):2}")

with open(CSV_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Category",
        "Train Images",
        "Test Images",
        "Defect Types",
        "Train Exists",
        "Test Exists",
        "Ground Truth Exists"
    ])

    writer.writerows(rows)

print("\nDone!")
print(f"CSV saved to:\n{CSV_FILE}")