"""
train_yolo_masks_v1.py
======================
Production YOLO11n training pipeline for the Industrial-AI-QC project.

Trains a YOLO11n object detector on the masks-derived dataset
(datasets_yolo_masks/) which covers 73 defect classes across 15 MVTec
categories.  Results feed into the hybrid YOLO + PatchCore inference
pipeline.

Dataset
-------
    datasets_yolo_masks/data.yaml
    - 73 defect classes  (category__defect naming)
    - 15 product categories
    - 7,912 train images  |  1,980 validation images

GPU target: NVIDIA GeForce RTX 2050 (4 GB VRAM)

Usage
-----
    python scripts/training/train_yolo_masks_v1.py
    python scripts/training/train_yolo_masks_v1.py --epochs 150 --batch 6
    python scripts/training/train_yolo_masks_v1.py --resume runs/detect/runs/masks/yolo11n_masks_v1/weights/last.pt

Outputs
-------
    runs/detect/runs/masks/yolo11n_masks_v1/
        weights/best.pt   weights/last.pt
        results.csv       confusion_matrix.png
        PR_curve.png      F1_curve.png
    models/checkpoints/best_masks_v1.pt     <- canonical checkpoint
    reports/yolo_masks_v1_report.md         <- human-readable summary

Author: Srinath Sampathkumar
Compatible with: ultralytics==8.4.x, PyTorch 2.5.x, Windows
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError in logs)
# ──────────────────────────────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch

# Unlock Tensor Core speed on RTX GPUs (medium precision, no accuracy impact)
torch.set_float32_matmul_precision("medium")

# ──────────────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────────────

# Script lives at scripts/training/  ->  project root is 2 levels up
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_YAML:   Path = PROJECT_ROOT / "datasets_yolo_masks" / "data.yaml"
WEIGHTS_PT:  Path = PROJECT_ROOT / "yolo11n.pt"          # pretrained backbone
RUNS_DIR:    Path = PROJECT_ROOT / "runs" / "detect" / "runs" / "masks"
RUN_NAME:    str  = "yolo11n_masks_v1"
CKPT_DIR:    Path = PROJECT_ROOT / "models" / "checkpoints"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters  (RTX 2050 / 4 GB VRAM – tuned for this GPU)
# ──────────────────────────────────────────────────────────────────────────────

HYPERPARAMS: dict = {
    # ── Architecture ──────────────────────────────────────────────────────────
    "imgsz":         640,       # Input resolution; YOLO11n was designed at 640
    # ── Compute ───────────────────────────────────────────────────────────────
    "batch":         8,         # Safe ceiling for 4 GB VRAM at 640px
    "workers":       4,         # Windows-compatible; 0 would be safe too
    "device":        0,         # GPU 0 (RTX 2050)
    "amp":           True,      # Mixed precision (AMP) – critical at 4 GB
    "cache":         "disk",    # Disk cache avoids OOM that RAM cache triggers
    # ── Schedule ──────────────────────────────────────────────────────────────
    "epochs":        100,       # Enough for convergence; patience will stop early
    "patience":      20,        # Early stopping after 20 stagnant epochs
    "warmup_epochs": 3,         # Linear LR warmup
    "warmup_momentum": 0.8,     # Warmup SGD momentum
    "warmup_bias_lr":  0.1,     # Warmup bias-specific LR
    # ── Learning rate ─────────────────────────────────────────────────────────
    "optimizer":     "AdamW",   # Better than SGD for fine-tuning on small data
    "lr0":           1e-3,      # Initial LR (AdamW is stable at this value)
    "lrf":           0.01,      # Final LR factor: lr_final = lr0 * lrf
    "momentum":      0.937,     # Adam beta1 mapped to momentum
    "weight_decay":  5e-4,      # L2 regularisation
    "cos_lr":        True,      # Cosine annealing LR schedule
    # ── Augmentation ──────────────────────────────────────────────────────────
    "hsv_h":         0.015,     # Hue jitter
    "hsv_s":         0.7,       # Saturation jitter
    "hsv_v":         0.4,       # Value (brightness) jitter
    "degrees":       5.0,       # Random rotation ±5°
    "translate":     0.1,       # Random translation
    "scale":         0.5,       # Random scale
    "shear":         2.0,       # Random shear ±2°
    "perspective":   0.0,       # No perspective (industrial textures)
    "flipud":        0.0,       # No vertical flip (orientation matters)
    "fliplr":        0.5,       # Horizontal flip 50%
    "mosaic":        1.0,       # Mosaic augmentation (helps with 73 classes)
    "mixup":         0.0,       # No mixup (hurts small defect localisation)
    "copy_paste":    0.0,       # No copy-paste (mask-derived bboxes only)
    # ── Loss weights ──────────────────────────────────────────────────────────
    "box":           7.5,       # Box regression loss weight
    "cls":           0.5,       # Classification loss weight
    "dfl":           1.5,       # Distribution focal loss weight
    # ── Reproducibility ───────────────────────────────────────────────────────
    "seed":          42,        # Fully deterministic training
    "deterministic": True,
    # ── Ultralytics project control ───────────────────────────────────────────
    "pretrained":    True,      # Start from yolo11n.pt ImageNet weights
    "verbose":       True,
    "exist_ok":      False,     # Never silently overwrite a previous run
    "plots":         True,      # Save all training curves and confusion matrix
    "save":          True,      # Save best.pt + last.pt
    "save_period":   -1,        # Only save best/last (disable periodic saves)
    "val":           True,      # Run validation at every epoch
}


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure a UTF-8-safe logger for Windows."""
    logger = logging.getLogger("train_yolo_masks_v1")
    if not logger.handlers:
        handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = setup_logging()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the most commonly tweaked settings."""
    parser = argparse.ArgumentParser(
        description="Production YOLO11n training on datasets_yolo_masks/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=int, default=HYPERPARAMS["epochs"],
        help="Total training epochs.",
    )
    parser.add_argument(
        "--batch", type=int, default=HYPERPARAMS["batch"],
        help="Batch size (images per step). Reduce to 6 if CUDA OOM occurs.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=HYPERPARAMS["imgsz"],
        help="Input image size (square).",
    )
    parser.add_argument(
        "--patience", type=int, default=HYPERPARAMS["patience"],
        help="Early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--lr0", type=float, default=HYPERPARAMS["lr0"],
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--workers", type=int, default=HYPERPARAMS["workers"],
        help="DataLoader worker processes. Set 0 if multiprocessing errors occur.",
    )
    parser.add_argument(
        "--device", default=str(HYPERPARAMS["device"]),
        help="Device string passed to Ultralytics: '0', 'cpu', '0,1', etc.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="LAST_PT",
        help="Path to last.pt to resume an interrupted run.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed precision (AMP). Use only if AMP causes issues.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict:
    """Return a dict with GPU name and VRAM in GB, or empty strings for CPU."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "name":      props.name,
            "vram_gb":   round(props.total_memory / 1e9, 2),
            "cuda_ver":  torch.version.cuda or "unknown",
            "torch_ver": torch.__version__,
        }
    return {"name": "CPU", "vram_gb": 0.0, "cuda_ver": "N/A", "torch_ver": torch.__version__}


def get_ultralytics_version() -> str:
    try:
        import ultralytics
        return ultralytics.__version__
    except Exception:
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ──────────────────────────────────────────────────────────────────────────────

def verify_environment(log: logging.Logger) -> None:
    """
    Validate that all required files and directories exist before training starts.
    Raises FileNotFoundError / RuntimeError on problems.
    """
    log.info("Running pre-flight checks ...")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML}\n"
            "Run scripts/dataset_tools/create_dataset_from_masks_v2.py first."
        )

    if not WEIGHTS_PT.exists():
        raise FileNotFoundError(
            f"Base weights not found: {WEIGHTS_PT}\n"
            "Download yolo11n.pt from https://github.com/ultralytics/assets/releases"
        )

    if not torch.cuda.is_available():
        log.warning("CUDA not available - training will run on CPU (very slow).")

    train_images = list((DATA_YAML.parent / "images" / "train").glob("*"))
    val_images   = list((DATA_YAML.parent / "images" / "val").glob("*"))
    log.info("  data.yaml      : %s", DATA_YAML)
    log.info("  base weights   : %s", WEIGHTS_PT)
    log.info("  train images   : %d", len(train_images))
    log.info("  val   images   : %d", len(val_images))

    if len(train_images) == 0:
        raise RuntimeError("No training images found under datasets_yolo_masks/images/train/")
    if len(val_images) == 0:
        raise RuntimeError("No validation images found under datasets_yolo_masks/images/val/")

    log.info("Pre-flight checks passed.")


# ──────────────────────────────────────────────────────────────────────────────
# Run directory resolution
# ──────────────────────────────────────────────────────────────────────────────

def resolve_run_dir() -> tuple[Path, str]:
    """
    Return (project_dir, run_name) for the Ultralytics YOLO trainer.

    YOLO saves to  <project>/<name>/  and auto-increments to <name>2 if the
    folder already exists (when exist_ok=False).

    We pass exist_ok=False so the Ultralytics engine itself handles the
    non-overwriting guarantee cleanly.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR, RUN_NAME


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def run_training(args: argparse.Namespace, log: logging.Logger) -> dict:
    """
    Execute YOLO11n training and return a results dict.

    Returns
    -------
    dict with keys:
        success, run_dir, best_pt, last_pt, train_time_s,
        metrics, error
    """
    from ultralytics import YOLO

    gpu_info = get_gpu_info()

    # ── Load model ──────────────────────────────────────────────────────────
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        log.info("Resuming from checkpoint: %s", resume_path)
        model = YOLO(str(resume_path))
        resuming = True
    else:
        model = YOLO(str(WEIGHTS_PT))
        resuming = False

    # ── Merge CLI overrides into hyperparams ────────────────────────────────
    hp = dict(HYPERPARAMS)  # shallow copy – don't mutate the module constant
    hp["epochs"]   = args.epochs
    hp["batch"]    = args.batch
    hp["imgsz"]    = args.imgsz
    hp["patience"] = args.patience
    hp["lr0"]      = args.lr0
    hp["workers"]  = args.workers
    hp["device"]   = args.device
    hp["amp"]      = not args.no_amp

    project_dir, run_name = resolve_run_dir()

    log.info("=" * 60)
    log.info("YOLO11n Training  |  masks dataset v1")
    log.info("=" * 60)
    log.info("  GPU            : %s (%.2f GB VRAM)", gpu_info["name"], gpu_info["vram_gb"])
    log.info("  CUDA           : %s", gpu_info["cuda_ver"])
    log.info("  PyTorch        : %s", gpu_info["torch_ver"])
    log.info("  Ultralytics    : %s", get_ultralytics_version())
    log.info("  data.yaml      : %s", DATA_YAML)
    log.info("  base weights   : %s", WEIGHTS_PT)
    log.info("  run directory  : %s/%s", project_dir, run_name)
    log.info("  Epochs         : %d", hp["epochs"])
    log.info("  Batch          : %d", hp["batch"])
    log.info("  Image size     : %d px", hp["imgsz"])
    log.info("  Optimizer      : %s", hp["optimizer"])
    log.info("  LR0            : %.4f", hp["lr0"])
    log.info("  Cosine LR      : %s", hp["cos_lr"])
    log.info("  AMP            : %s", hp["amp"])
    log.info("  Patience       : %d", hp["patience"])
    log.info("  Seed           : %d", hp["seed"])
    log.info("  Cache          : %s", hp["cache"])
    log.info("=" * 60)

    result: dict = {
        "success":       False,
        "run_dir":       None,
        "best_pt":       None,
        "last_pt":       None,
        "train_time_s":  0.0,
        "metrics":       {},
        "error":         "",
        "gpu_info":      gpu_info,
        "hyperparams":   hp,
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "finished_at":   None,
    }

    start_time = time.time()

    try:
        torch.cuda.empty_cache()

        # ── Train ───────────────────────────────────────────────────────────
        train_results = model.train(
            data     = str(DATA_YAML),
            project  = str(project_dir),
            name     = run_name,
            resume   = resuming,
            # -- Compute
            epochs   = hp["epochs"],
            batch    = hp["batch"],
            imgsz    = hp["imgsz"],
            device   = hp["device"],
            workers  = hp["workers"],
            amp      = hp["amp"],
            cache    = hp["cache"],
            # -- Schedule
            patience          = hp["patience"],
            warmup_epochs     = hp["warmup_epochs"],
            warmup_momentum   = hp["warmup_momentum"],
            warmup_bias_lr    = hp["warmup_bias_lr"],
            # -- Optimiser
            optimizer    = hp["optimizer"],
            lr0          = hp["lr0"],
            lrf          = hp["lrf"],
            momentum     = hp["momentum"],
            weight_decay = hp["weight_decay"],
            cos_lr       = hp["cos_lr"],
            # -- Augmentation
            hsv_h       = hp["hsv_h"],
            hsv_s       = hp["hsv_s"],
            hsv_v       = hp["hsv_v"],
            degrees     = hp["degrees"],
            translate   = hp["translate"],
            scale       = hp["scale"],
            shear       = hp["shear"],
            perspective = hp["perspective"],
            flipud      = hp["flipud"],
            fliplr      = hp["fliplr"],
            mosaic      = hp["mosaic"],
            mixup       = hp["mixup"],
            copy_paste  = hp["copy_paste"],
            # -- Loss weights
            box = hp["box"],
            cls = hp["cls"],
            dfl = hp["dfl"],
            # -- Reproducibility
            seed          = hp["seed"],
            deterministic = hp["deterministic"],
            # -- Misc
            pretrained  = hp["pretrained"],
            verbose     = hp["verbose"],
            exist_ok    = hp["exist_ok"],
            plots       = hp["plots"],
            save        = hp["save"],
            save_period = hp["save_period"],
            val         = hp["val"],
        )

        elapsed = time.time() - start_time
        result["train_time_s"] = round(elapsed, 1)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()

        # ── Locate run directory ─────────────────────────────────────────────
        # Ultralytics may auto-increment the name (e.g. yolo11n_masks_v12)
        # when exist_ok=False. We find the newest matching directory.
        run_dir = _find_run_dir(project_dir, run_name)
        result["run_dir"] = str(run_dir)

        best_pt = run_dir / "weights" / "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        result["best_pt"] = str(best_pt) if best_pt.exists() else None
        result["last_pt"] = str(last_pt) if last_pt.exists() else None

        log.info("Training finished in %.1f s (%.1f min).", elapsed, elapsed / 60)
        log.info("Run directory : %s", run_dir)
        log.info("best.pt       : %s", best_pt)
        log.info("last.pt       : %s", last_pt)

        result["success"] = True

    except Exception as exc:
        elapsed = time.time() - start_time
        result["train_time_s"] = round(elapsed, 1)
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["error"] = str(exc)
        LOG.error("Training failed: %s", exc, exc_info=True)

    return result


def _find_run_dir(project_dir: Path, run_name: str) -> Path:
    """
    Find the run directory Ultralytics created.

    YOLO11 may append a numeric suffix (yolo11n_masks_v12, yolo11n_masks_v13,
    ...) when exist_ok=False.  We pick the most recently modified directory
    whose name starts with run_name.
    """
    candidates = sorted(
        [d for d in project_dir.iterdir()
         if d.is_dir() and d.name.startswith(run_name)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # Fallback: exact name
    fallback = project_dir / run_name
    if fallback.is_dir():
        return fallback

    raise RuntimeError(
        f"Cannot locate the training run directory under {project_dir}. "
        f"Expected a folder starting with '{run_name}'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Validation on best.pt
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(best_pt: str, log: logging.Logger) -> dict:
    """
    Run model.val() on best.pt and return the metrics dict.

    Returns an empty dict on failure (training results are still kept).
    """
    from ultralytics import YOLO

    log.info("=" * 60)
    log.info("Running post-training validation on best.pt ...")
    log.info("  weights : %s", best_pt)
    log.info("  data    : %s", DATA_YAML)
    log.info("=" * 60)

    try:
        model = YOLO(best_pt)
        val_results = model.val(
            data    = str(DATA_YAML),
            imgsz   = HYPERPARAMS["imgsz"],
            batch   = HYPERPARAMS["batch"],
            device  = HYPERPARAMS["device"],
            workers = HYPERPARAMS["workers"],
            verbose = True,
            plots   = True,
        )

        # Extract scalar metrics from the results object
        metrics = _extract_metrics(val_results)

        log.info("Validation complete.")
        log.info("  Precision   (P)    : %.4f", metrics.get("precision", 0.0))
        log.info("  Recall      (R)    : %.4f", metrics.get("recall", 0.0))
        log.info("  mAP@50             : %.4f", metrics.get("map50", 0.0))
        log.info("  mAP@50-95          : %.4f", metrics.get("map50_95", 0.0))

        return metrics

    except Exception as exc:
        log.warning("Post-training validation failed: %s", exc, exc_info=True)
        return {}


def _extract_metrics(val_results) -> dict:
    """
    Safely extract scalar metrics from an Ultralytics Results object.

    The Ultralytics API stores metrics in val_results.results_dict.
    Keys are like 'metrics/precision(B)', 'metrics/recall(B)', etc.
    We normalise to simple flat keys.
    """
    metrics: dict = {}

    # Strategy 1: results_dict (most common in ultralytics 8.x)
    try:
        rd = val_results.results_dict
        mapping = {
            "metrics/precision(B)": "precision",
            "metrics/recall(B)":    "recall",
            "metrics/mAP50(B)":     "map50",
            "metrics/mAP50-95(B)":  "map50_95",
        }
        for src, dst in mapping.items():
            if src in rd:
                metrics[dst] = float(rd[src])
        if metrics:
            return metrics
    except Exception:
        pass

    # Strategy 2: box attribute (older API)
    try:
        box = val_results.box
        metrics = {
            "precision": float(box.mp),
            "recall":    float(box.mr),
            "map50":     float(box.map50),
            "map50_95":  float(box.map),
        }
        return metrics
    except Exception:
        pass

    # Strategy 3: attribute scan
    try:
        for attr in ("mp", "mr", "map50", "map"):
            val = getattr(val_results, attr, None)
            if val is not None:
                metrics[attr] = float(val)
    except Exception:
        pass

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint copy
# ──────────────────────────────────────────────────────────────────────────────

def copy_best_checkpoint(best_pt: str, log: logging.Logger) -> Optional[Path]:
    """
    Copy best.pt to models/checkpoints/best_masks_v1.pt.

    Creates the directory if it does not exist.
    Returns the destination path on success, None on failure.
    """
    try:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        dst = CKPT_DIR / "best_masks_v1.pt"
        shutil.copy2(best_pt, dst)
        log.info("Checkpoint copied: %s -> %s", best_pt, dst)
        return dst
    except Exception as exc:
        log.warning("Could not copy checkpoint: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report
# ──────────────────────────────────────────────────────────────────────────────

def write_markdown_report(
    result: dict,
    val_metrics: dict,
    checkpoint_dst: Optional[Path],
) -> Path:
    """
    Generate reports/yolo_masks_v1_report.md with all training details.

    Returns the path to the written report.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "yolo_masks_v1_report.md"

    gpu = result.get("gpu_info", {})
    hp  = result.get("hyperparams", {})
    elapsed_s = result.get("train_time_s", 0.0)
    elapsed_h = elapsed_s / 3600
    elapsed_m = elapsed_s / 60

    # Dataset stats (read from data.yaml path)
    try:
        train_count = len(list((DATA_YAML.parent / "images" / "train").glob("*")))
        val_count   = len(list((DATA_YAML.parent / "images" / "val").glob("*")))
    except Exception:
        train_count = val_count = "?"

    lines = [
        "# YOLO11n Masks v1 — Training Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> Status: {'SUCCESS' if result['success'] else 'FAILED'}",
        "",
        "---",
        "",
        "## Dataset Statistics",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `datasets_yolo_masks/` |",
        f"| data.yaml | `{DATA_YAML}` |",
        f"| Number of Categories | 15 |",
        f"| Number of Classes | 73 |",
        f"| Train Images | {train_count:,} |",
        f"| Validation Images | {val_count:,} |",
        f"| Total Images | {int(train_count) + int(val_count):,} |",
        "",
        "---",
        "",
        "## Hyperparameters",
        "",
        "| Hyperparameter | Value |",
        "|---|---|",
        f"| Model | YOLO11n |",
        f"| Base Weights | `yolo11n.pt` (ImageNet pretrained) |",
        f"| Image Size | {hp.get('imgsz', '?')} px |",
        f"| Epochs | {hp.get('epochs', '?')} |",
        f"| Batch Size | {hp.get('batch', '?')} |",
        f"| Optimizer | {hp.get('optimizer', '?')} |",
        f"| Learning Rate (LR0) | {hp.get('lr0', '?')} |",
        f"| Final LR Factor (LRF) | {hp.get('lrf', '?')} |",
        f"| Cosine LR | {hp.get('cos_lr', '?')} |",
        f"| Weight Decay | {hp.get('weight_decay', '?')} |",
        f"| Warmup Epochs | {hp.get('warmup_epochs', '?')} |",
        f"| Patience (early stop) | {hp.get('patience', '?')} |",
        f"| Mixed Precision (AMP) | {hp.get('amp', '?')} |",
        f"| Cache | {hp.get('cache', '?')} |",
        f"| Workers | {hp.get('workers', '?')} |",
        f"| Mosaic Augmentation | {hp.get('mosaic', '?')} |",
        f"| Horizontal Flip | {hp.get('fliplr', '?')} |",
        f"| Random Scale | {hp.get('scale', '?')} |",
        f"| Seed | {hp.get('seed', '?')} |",
        f"| Deterministic | {hp.get('deterministic', '?')} |",
        "",
        "---",
        "",
        "## Hardware",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {gpu.get('name', 'N/A')} |",
        f"| VRAM | {gpu.get('vram_gb', 'N/A')} GB |",
        f"| CUDA | {gpu.get('cuda_ver', 'N/A')} |",
        f"| PyTorch | {gpu.get('torch_ver', 'N/A')} |",
        f"| Ultralytics | {get_ultralytics_version()} |",
        "",
        "---",
        "",
        "## Training Duration",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Started at | {result.get('started_at', 'N/A')} |",
        f"| Finished at | {result.get('finished_at', 'N/A')} |",
        f"| Total Training Time | {elapsed_s:.1f} s  ({elapsed_m:.1f} min / {elapsed_h:.2f} h) |",
        "",
        "---",
        "",
        "## Final Metrics (Post-Training Validation on best.pt)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision (P) | {val_metrics.get('precision', 'N/A'):.4f} |"
        if isinstance(val_metrics.get('precision'), float) else
        f"| Precision (P) | N/A |",
        f"| Recall (R) | {val_metrics.get('recall', 'N/A'):.4f} |"
        if isinstance(val_metrics.get('recall'), float) else
        f"| Recall (R) | N/A |",
        f"| mAP@50 | {val_metrics.get('map50', 'N/A'):.4f} |"
        if isinstance(val_metrics.get('map50'), float) else
        f"| mAP@50 | N/A |",
        f"| mAP@50-95 | {val_metrics.get('map50_95', 'N/A'):.4f} |"
        if isinstance(val_metrics.get('map50_95'), float) else
        f"| mAP@50-95 | N/A |",
        "",
        "---",
        "",
        "## Output Locations",
        "",
        "| Artifact | Path |",
        "|---|---|",
        f"| Run directory | `{result.get('run_dir', 'N/A')}` |",
        f"| best.pt | `{result.get('best_pt', 'N/A')}` |",
        f"| last.pt | `{result.get('last_pt', 'N/A')}` |",
        f"| Canonical checkpoint | `{checkpoint_dst or 'N/A'}` |",
        f"| This report | `{report_path}` |",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- All 73 defect classes follow the `category__defect` naming convention.",
        "- Good/background images are included with empty label files (YOLO negative samples).",
        "- This model is intended to be combined with the trained PatchCore models",
        "  for a hybrid YOLO + PatchCore industrial defect detection pipeline.",
        "",
    ]

    if not result["success"] and result.get("error"):
        lines += [
            "## Error Details",
            "",
            "```",
            result["error"],
            "```",
            "",
        ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ──────────────────────────────────────────────────────────────────────────────
# Terminal summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(result: dict, val_metrics: dict, checkpoint_dst: Optional[Path]) -> None:
    """Print a clean ASCII summary table to stdout after training."""
    elapsed_s = result.get("train_time_s", 0.0)

    print()
    print("=" * 72)
    print(f"  YOLO11n MASKS v1  |  TRAINING SUMMARY")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    status = "SUCCESS" if result["success"] else "FAILED"
    print(f"  Status             : {status}")
    print(f"  Training Time      : {elapsed_s:.1f} s  ({elapsed_s/60:.1f} min)")

    # Metrics
    print()
    print("  Final Metrics (best.pt):")
    print(f"  {'Precision (P)':<22}: "
          f"{val_metrics.get('precision', 0.0):.4f}"
          if isinstance(val_metrics.get("precision"), float)
          else f"  {'Precision (P)':<22}: N/A")
    print(f"  {'Recall    (R)':<22}: "
          f"{val_metrics.get('recall', 0.0):.4f}"
          if isinstance(val_metrics.get("recall"), float)
          else f"  {'Recall    (R)':<22}: N/A")
    print(f"  {'mAP@50':<22}: "
          f"{val_metrics.get('map50', 0.0):.4f}"
          if isinstance(val_metrics.get("map50"), float)
          else f"  {'mAP@50':<22}: N/A")
    print(f"  {'mAP@50-95':<22}: "
          f"{val_metrics.get('map50_95', 0.0):.4f}"
          if isinstance(val_metrics.get("map50_95"), float)
          else f"  {'mAP@50-95':<22}: N/A")

    print()
    print("  Output Locations:")
    print(f"  {'Run directory':<22}: {result.get('run_dir', 'N/A')}")
    print(f"  {'best.pt':<22}: {result.get('best_pt', 'N/A')}")
    print(f"  {'last.pt':<22}: {result.get('last_pt', 'N/A')}")
    print(f"  {'Checkpoint copy':<22}: {checkpoint_dst or 'N/A'}")

    if result.get("error"):
        print()
        print(f"  Error: {result['error'][:120]}")

    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Orchestrate the full training pipeline:

    1. Parse CLI args
    2. Pre-flight checks
    3. Training
    4. Post-training validation on best.pt
    5. Copy best.pt to models/checkpoints/
    6. Write markdown report
    7. Print summary
    8. Return exit code (0 = success, 1 = training failed, 2 = val failed)
    """
    args = parse_args()

    LOG.info("=" * 60)
    LOG.info("Industrial-AI-QC  |  YOLO11n Masks v1 Training Pipeline")
    LOG.info("=" * 60)
    LOG.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ── Pre-flight ───────────────────────────────────────────────────────────
    try:
        verify_environment(LOG)
    except (FileNotFoundError, RuntimeError) as exc:
        LOG.error("Pre-flight check failed: %s", exc)
        return 1

    # ── Ensure output directories exist ─────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG.info("models/checkpoints/ ready: %s", CKPT_DIR)

    # ── Training ─────────────────────────────────────────────────────────────
    result = run_training(args, LOG)

    # ── Post-training validation ──────────────────────────────────────────────
    val_metrics: dict = {}
    exit_code = 0

    if result["success"] and result["best_pt"]:
        val_metrics = run_validation(result["best_pt"], LOG)
        if not val_metrics:
            LOG.warning("Post-training validation returned no metrics.")
            exit_code = 2
    elif not result["success"]:
        LOG.error("Training failed — skipping validation.")
        exit_code = 1

    # ── Copy checkpoint ───────────────────────────────────────────────────────
    checkpoint_dst: Optional[Path] = None
    if result["success"] and result["best_pt"]:
        checkpoint_dst = copy_best_checkpoint(result["best_pt"], LOG)

    # ── Markdown report ───────────────────────────────────────────────────────
    try:
        report_path = write_markdown_report(result, val_metrics, checkpoint_dst)
        LOG.info("Report saved: %s", report_path)
    except Exception as exc:
        LOG.warning("Could not write markdown report: %s", exc)
        report_path = None

    # ── JSON artifact (machine-readable) ──────────────────────────────────────
    try:
        artifact = {
            "training": result,
            "validation_metrics": val_metrics,
            "checkpoint_copy": str(checkpoint_dst) if checkpoint_dst else None,
            "report_path": str(report_path) if report_path else None,
        }
        json_path = REPORTS_DIR / "yolo_masks_v1_result.json"
        json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        LOG.info("JSON artifact saved: %s", json_path)
    except Exception as exc:
        LOG.warning("Could not save JSON artifact: %s", exc)

    # ── Terminal summary ──────────────────────────────────────────────────────
    print_summary(result, val_metrics, checkpoint_dst)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
