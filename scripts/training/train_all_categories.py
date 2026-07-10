"""
Multi-Category PatchCore Batch Training Script
===============================================
Automatically discovers all MVTec AD categories in datasets/ and trains
a PatchCore model for each one.

Usage:
    python scripts/training/train_all_categories.py
    python scripts/training/train_all_categories.py --categories bottle cable capsule
    python scripts/training/train_all_categories.py --coreset-ratio 0.01 --batch-size 32

Features:
    - Auto-discovers all categories in datasets/
    - Continues training if one category fails (never stops)
    - tqdm progress bar with live status
    - Generates reports/training/training_log.csv
    - Compatible with anomalib==2.5.0

Compatible with: anomalib==2.5.0, PyTorch 2.5.x, Windows
"""

import argparse
import csv
import io
import os
import sys
import time
from datetime import datetime

# Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
MODELS_DIR   = PROJECT_ROOT / "models" / "patchcore"
REPORTS_DIR  = PROJECT_ROOT / "reports" / "training"

# ─────────────────────────────────────────────────────────────
# Import the single-category trainer (sibling module)
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_patchcore import train_category   # noqa: E402

# ─────────────────────────────────────────────────────────────
# All known MVTec AD categories (canonical order)
# ─────────────────────────────────────────────────────────────
MVTEC_CATEGORIES = [
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


def discover_categories(dataset_root: Path) -> list[str]:
    """
    Discover all valid MVTec categories present in dataset_root.
    A valid category must have datasets/<name>/train/good/ directory.
    Returns list of category names in canonical MVTEC_CATEGORIES order.
    """
    found = []
    for cat in MVTEC_CATEGORIES:
        train_good = dataset_root / cat / "train" / "good"
        if train_good.exists() and any(train_good.glob("*.*")):
            found.append(cat)

    # Also pick up any extra categories not in MVTEC_CATEGORIES
    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue
        cat = cat_dir.name
        if cat in MVTEC_CATEGORIES:
            continue  # already handled
        train_good = cat_dir / "train" / "good"
        if train_good.exists() and any(train_good.glob("*.*")):
            found.append(cat)

    return found


def save_training_log(results: list[dict], log_path: Path):
    """Write training_log.csv with all results."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "category",
        "status",
        "training_images",
        "training_time_s",
        "error",
        "output_dir",
    ]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def print_summary(results: list[dict], total_time: float):
    """Print a clean summary table after all categories finish."""
    successes = [r for r in results if r["status"] == "SUCCESS"]
    failures  = [r for r in results if r["status"] != "SUCCESS"]

    print()
    print("=" * 72)
    print(f"  TRAINING SUMMARY  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    print(f"  {'Category':<14} {'Status':<10} {'Images':>8} {'Time (s)':>10}  Error")
    print("-" * 72)

    for r in results:
        status_icon = "OK" if r["status"] == "SUCCESS" else "!!"
        err_preview = (r.get("error", "") or "")[:30]
        print(f"  [{status_icon}] {r['category']:<13} {r['status']:<10} "
              f"{r.get('training_images', 0):>8} {r.get('training_time_s', 0):>10.1f}"
              f"  {err_preview}")

    print("-" * 72)
    print(f"  Total: {len(results)} categories | "
          f"Success: {len(successes)} | "
          f"Failed: {len(failures)} | "
          f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore for all MVTec AD categories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to train (default: all discovered in datasets/)",
        metavar="CATEGORY",
    )
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        help="Timm backbone for feature extraction",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        help="Backbone layers to extract features from",
    )
    parser.add_argument(
        "--coreset-ratio",
        type=float,
        default=0.01,
        dest="coreset_ratio",
        help="Coreset sampling ratio (0.01 = 1%%)",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        dest="num_neighbors",
        help="Number of nearest neighbors for anomaly scoring",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        dest="batch_size",
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--log-csv",
        default=str(REPORTS_DIR / "training_log.csv"),
        dest="log_csv",
        help="Path for the output CSV training log",
    )
    args = parser.parse_args()

    # ── Determine categories to train ─────────────────────────
    if args.categories:
        categories = args.categories
        print(f"Training specified categories: {categories}")
    else:
        categories = discover_categories(DATASET_ROOT)
        print(f"Auto-discovered {len(categories)} categories: {categories}")

    if not categories:
        print("ERROR: No valid categories found. Check that datasets/ contains "
              "category folders with train/good/ subdirectories.")
        sys.exit(1)

    log_path = Path(args.log_csv)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"  PatchCore Multi-Category Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Categories: {len(categories)}")
    print(f"  Backbone: {args.backbone}  |  Coreset: {args.coreset_ratio:.1%}")
    print(f"  Output: {MODELS_DIR}")
    print(f"  Log: {log_path}")
    print("=" * 72)
    print()

    # ── Import tqdm (graceful fallback if not installed) ──────
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("tqdm not found - running without progress bar.")
        use_tqdm = False

    results       = []
    all_start     = time.time()

    iterator = (
        tqdm(categories, desc="Training categories", unit="cat", ncols=80)
        if use_tqdm else categories
    )

    for category in iterator:
        if use_tqdm:
            iterator.set_description(f"Training: {category:<12}")

        print()
        print("-" * 72)
        print(f"  [{len(results)+1}/{len(categories)}] Category: {category.upper()}")
        print("-" * 72)

        result = train_category(
            category=category,
            backbone=args.backbone,
            layers=args.layers,
            coreset_ratio=args.coreset_ratio,
            num_neighbors=args.num_neighbors,
            batch_size=args.batch_size,
        )
        results.append(result)

        # Save CSV after every category (in case of crash, partial results preserved)
        save_training_log(results, log_path)

        status_str = ("[OK] SUCCESS" if result["status"] == "SUCCESS"
                      else f"[!!] FAILED: {result.get('error','')[:60]}")
        print(f"  → {category}: {status_str}  ({result['training_time_s']:.1f}s)")

    # ── Final report ──────────────────────────────────────────
    total_time = time.time() - all_start
    print_summary(results, total_time)

    # Append summary row
    save_training_log(results, log_path)
    print(f"\n  Training log saved: {log_path}")

    # Exit with failure code if any category failed
    failures = [r for r in results if r["status"] != "SUCCESS"]
    if failures:
        print(f"\nWARNING: {len(failures)} categories failed training.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} categories trained successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
