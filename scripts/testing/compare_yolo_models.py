"""
YOLO Model Comparison Script
=============================

Evaluates both YOLO models on the same validation set and produces a
side-by-side metrics report.

Models evaluated
----------------
  pretrained  →  yolo11n.pt             (COCO-pretrained, 80 classes)
  finetuned   →  runs/detect/runs/msme_defect_detection-3/weights/best.pt
                                         (fine-tuned, 10 MSME defect classes)

Dataset
-------
  roboflow_dataset/data.yaml
  val split: roboflow_dataset/valid/images/
  Classes  : Color, Crack, Cut, Fold, Glue, Glue_Strip, Gray_Stroke, Oil, Poke, Rough

  This is the exact dataset the fine-tuned model was trained on, making it
  the correct benchmark surface for both models.  The pretrained model will
  score near-zero (it was trained on COCO objects, not industrial defects),
  which is the expected and meaningful baseline.

Metrics reported (per model)
----------------------------
  - Precision (B)
  - Recall    (B)
  - mAP50     (B)
  - mAP50-95  (B)

Output
------
  reports/yolo_model_comparison.csv

Usage
-----
  python scripts/testing/compare_yolo_models.py
  python scripts/testing/compare_yolo_models.py --imgsz 640 --conf 0.001 --iou 0.6

Author: Srinath
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Paths — sourced from central config
# ------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "yolo_models.yaml"
# roboflow_dataset is the training+validation dataset for the fine-tuned model.
# Using the same dataset for both models gives a valid apples-to-apples comparison.
DATA_YAML = PROJECT_ROOT / "roboflow_dataset" / "data.yaml"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_CSV = REPORTS_DIR / "yolo_model_comparison.csv"


def load_model_paths() -> dict[str, str]:
    """Read pretrained/finetuned weight paths from configs/yolo_models.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "pretrained": str(PROJECT_ROOT / cfg["pretrained"]),
        "finetuned": str(PROJECT_ROOT / cfg["finetuned"]),
    }


# ------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------

def build_resolved_data_yaml(data_yaml: Path, tmp_dir: Path) -> Path:
    """
    Ultralytics resolves data.yaml paths relative to the *process working
    directory*, not the yaml file's own directory.  This breaks when the two
    differ.

    This function writes a one-shot resolved copy of data.yaml to a scratch
    directory with all paths made absolute.  The original yaml is NEVER modified.

    Handles two common layouts:
      - MVTec-style : has ``path: .`` and relative ``train/val`` under that root
      - Roboflow-style: no ``path`` key; ``train/val/test`` are direct relative
                        paths like ``../train/images``
    """
    with data_yaml.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = data_yaml.parent.resolve()

    if "path" in cfg:
        # MVTec-style: resolve path root relative to the yaml's own directory
        dataset_root = (yaml_dir / cfg["path"]).resolve()
        cfg["path"] = str(dataset_root)
    else:
        # Roboflow-style: no path key — resolve each split path individually.
        # The Roboflow yaml uses paths like '../train/images' designed to be
        # run from within a sub-directory.  We first try resolving relative to
        # the yaml's own directory; if that path doesn't exist we try resolving
        # the FINAL segment directly under the yaml directory instead.
        for split_key in ("train", "val", "test"):
            if split_key in cfg:
                split_val = cfg[split_key]
                if isinstance(split_val, str):
                    candidate = (yaml_dir / split_val).resolve()
                    if not candidate.exists():
                        # Fallback: try path segment directly under yaml_dir
                        stem = Path(split_val).name  # e.g. "images"
                        parent_stem = Path(split_val).parent.name  # e.g. "valid"
                        fallback = (yaml_dir / parent_stem / stem).resolve()
                        if fallback.exists():
                            candidate = fallback
                    cfg[split_key] = str(candidate)
                elif isinstance(split_val, list):
                    resolved_list = []
                    for p in split_val:
                        candidate = (yaml_dir / p).resolve()
                        if not candidate.exists():
                            stem = Path(p).name
                            parent_stem = Path(p).parent.name
                            fallback = (yaml_dir / parent_stem / stem).resolve()
                            if fallback.exists():
                                candidate = fallback
                        resolved_list.append(str(candidate))
                    cfg[split_key] = resolved_list

    tmp_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = tmp_dir / "data_resolved.yaml"
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    return resolved_yaml


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(
    label: str,
    weights_path: str,
    resolved_data_yaml: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict:
    """
    Run model.val() and extract the four required metrics.

    Returns a dict with keys:
        model, weights, precision, recall, map50, map50_95, eval_time_s
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Weights   : {weights_path}")
    print(f"{'='*60}")

    weights_p = Path(weights_path)
    if not weights_p.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_p))

    t_start = time.perf_counter()

    metrics = model.val(
        data=str(resolved_data_yaml),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        plots=False,
        save=False,
    )

    elapsed = time.perf_counter() - t_start

    # Ultralytics stores box metrics in metrics.box
    box = metrics.box

    precision   = float(box.mp)        # mean Precision
    recall      = float(box.mr)        # mean Recall
    map50       = float(box.map50)     # mAP@0.5
    map50_95    = float(box.map)       # mAP@0.5:0.95

    row = {
        "model":        label,
        "weights":      weights_path,
        "precision":    round(precision,  4),
        "recall":       round(recall,     4),
        "mAP50":        round(map50,      4),
        "mAP50-95":     round(map50_95,   4),
        "eval_time_s":  round(elapsed,    2),
    }

    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  mAP50      : {map50:.4f}")
    print(f"  mAP50-95   : {map50_95:.4f}")
    print(f"  Time       : {elapsed:.2f}s")

    return row



# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("YOLO MODEL COMPARISON — RESULTS")
    print("=" * 60)
    cols = ["model", "precision", "recall", "mAP50", "mAP50-95", "eval_time_s"]
    print(df[cols].to_string(index=False))
    print("=" * 60)

    # Delta row (finetuned − pretrained)
    if len(df) == 2:
        metric_cols = ["precision", "recall", "mAP50", "mAP50-95"]
        pretrained_row = df[df["model"] == "pretrained"][metric_cols].iloc[0]
        finetuned_row  = df[df["model"] == "finetuned"][metric_cols].iloc[0]
        delta = (finetuned_row - pretrained_row).round(4)
        print("\nDelta (finetuned − pretrained):")
        for col in metric_cols:
            sign = "+" if delta[col] >= 0 else ""
            print(f"  {col:<12}: {sign}{delta[col]:.4f}")
        print("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained vs fine-tuned YOLO models on the MSME validation set."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for val (default: 0.001 — standard for mAP calculation)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (default: 0.6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("YOLO Model Comparison")
    print(f"Dataset : {DATA_YAML}")
    print(f"Output  : {args.output}")
    print(f"imgsz={args.imgsz}  conf={args.conf}  iou={args.iou}")

    # Validate dataset config exists
    if not DATA_YAML.exists():
        print(f"ERROR: Dataset config not found: {DATA_YAML}", file=sys.stderr)
        return 1

    # Load weight paths
    try:
        model_paths = load_model_paths()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results = []

    # Build a resolved copy of data.yaml with absolute paths so Ultralytics
    # can find the validation images regardless of the working directory.
    tmp_dir = PROJECT_ROOT / ".tmp_yolo_compare"
    resolved_data_yaml = build_resolved_data_yaml(DATA_YAML, tmp_dir)
    print(f"Resolved dataset YAML : {resolved_data_yaml}")

    try:
        for label, weights in model_paths.items():
            try:
                row = evaluate_model(
                    label=label,
                    weights_path=weights,
                    resolved_data_yaml=resolved_data_yaml,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                )
                results.append(row)
            except FileNotFoundError as exc:
                print(f"SKIP {label}: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"ERROR evaluating {label}: {exc}", file=sys.stderr)
                raise
    finally:
        # Clean up the temporary resolved yaml
        if resolved_data_yaml.exists():
            resolved_data_yaml.unlink()
        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass  # not empty — leave it

    if not results:
        print("No models could be evaluated. Aborting.", file=sys.stderr)
        return 1

    df = pd.DataFrame(results)

    # Ensure consistent column order
    col_order = ["model", "weights", "precision", "recall", "mAP50", "mAP50-95", "eval_time_s"]
    df = df[[c for c in col_order if c in df.columns]]

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print_comparison_table(df)

    print(f"\nReport saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

