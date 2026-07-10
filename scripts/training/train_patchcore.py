"""
PatchCore Multi-Category Training Script
=========================================
Train a PatchCore anomaly detection model for ONE MVTec AD category.

Usage:
    python scripts/training/train_patchcore.py --category bottle
    python scripts/training/train_patchcore.py --category leather --coreset-ratio 0.01

Compatible with: anomalib==2.5.0, PyTorch 2.5.x, Windows

Outputs saved to: models/patchcore/<category>/
    ├── weights/              <- PyTorch Lightning checkpoint (.ckpt)
    ├── model.pt              <- Inference-ready state dict + memory bank
    ├── threshold.json        <- Anomaly threshold value
    ├── metrics.json          <- Final evaluation metrics
    └── metadata.json         <- Training metadata (time, images, config)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────
# Tensor Core precision (RTX GPUs)
# ─────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("medium")

# ─────────────────────────────────────────────────────────────
# Project paths (script lives in scripts/training/ → root is 2 levels up)
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
MODELS_DIR   = PROJECT_ROOT / "models" / "patchcore"


def setup_logging(category: str) -> logging.Logger:
    """Configure UTF-8 safe logging for Windows."""
    logger = logging.getLogger(f"train_patchcore.{category}")
    if not logger.handlers:
        handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def get_accelerator() -> tuple[str, int]:
    """Detect CUDA availability and return (accelerator, devices)."""
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 1


def verify_dataset(category: str, log: logging.Logger) -> Path:
    """Verify that the MVTec dataset for the given category is present."""
    category_dir = DATASET_ROOT / category
    train_good   = category_dir / "train" / "good"

    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")
    if not train_good.exists():
        raise FileNotFoundError(f"train/good not found: {train_good}")

    n_train = len(list(train_good.glob("*.*")))
    log.info("Dataset path   : %s", category_dir)
    log.info("Train images   : %d", n_train)

    if n_train == 0:
        raise RuntimeError(f"No training images found in {train_good}")

    return category_dir, n_train


def extract_threshold(engine, model, log: logging.Logger):
    """
    Extract the anomaly threshold from the trained model.
    Anomalib 2.5.0 stores it in engine.trainer.model.post_processor.
    Falls back gracefully if not accessible.
    """
    threshold_value = None

    # Strategy 1: post_processor on the Lightning wrapper
    try:
        trained_model = engine.trainer.model
        if hasattr(trained_model, "post_processor"):
            pp = trained_model.post_processor
            if hasattr(pp, "image_threshold"):
                threshold_value = float(pp.image_threshold.value)
                log.info("Threshold (post_processor.image_threshold): %.6f", threshold_value)
                return threshold_value
            elif hasattr(pp, "threshold"):
                threshold_value = float(pp.threshold)
                log.info("Threshold (post_processor.threshold): %.6f", threshold_value)
                return threshold_value
    except Exception as e:
        log.debug("Strategy 1 failed: %s", e)

    # Strategy 2: directly on the anomalib model
    try:
        if hasattr(model, "image_threshold"):
            threshold_value = float(model.image_threshold.value)
            log.info("Threshold (model.image_threshold): %.6f", threshold_value)
            return threshold_value
    except Exception as e:
        log.debug("Strategy 2 failed: %s", e)

    # Strategy 3: try normalization stats
    try:
        trained_model = engine.trainer.model
        if hasattr(trained_model, "normalization_metrics"):
            nm = trained_model.normalization_metrics
            log.debug("Normalization metrics: %s", nm)
    except Exception as e:
        log.debug("Strategy 3 failed: %s", e)

    log.warning("Could not extract threshold - will be fitted at inference time.")
    return None


def save_model_artifacts(model, engine, output_dir: Path, config: dict, log: logging.Logger):
    """Save all artifacts needed for later inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── model.pt (state dict + memory bank) ──────────────────
    model_pt_path = output_dir / "model.pt"
    try:
        save_dict = {
            "model_state_dict":      model.state_dict(),
            "backbone":              config["backbone"],
            "layers":                config["layers"],
            "num_neighbors":         config["num_neighbors"],
            "coreset_sampling_ratio": config["coreset_sampling_ratio"],
            "category":              config["category"],
        }
        # Include memory bank if accessible
        try:
            save_dict["memory_bank"] = model.model.memory_bank
            log.info("Memory bank shape  : %s", model.model.memory_bank.shape)
        except Exception:
            log.warning("Could not access model.model.memory_bank - skipping in model.pt")

        torch.save(save_dict, model_pt_path)
        log.info("model.pt saved     : %s", model_pt_path)
    except Exception as e:
        log.warning("Could not save model.pt: %s", e)

    # ── threshold.json ────────────────────────────────────────
    threshold_value = extract_threshold(engine, model, log)
    threshold_data = {
        "image_threshold": threshold_value if threshold_value is not None else "fitted_at_inference",
        "note": "PatchCore threshold fitted from validation scores during training.",
        "category": config["category"],
    }
    threshold_path = output_dir / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    log.info("threshold.json saved: %s", threshold_path)

    return threshold_value


def train_category(
    category: str,
    backbone: str = "wide_resnet50_2",
    layers: list = None,
    coreset_ratio: float = 0.01,
    num_neighbors: int = 9,
    batch_size: int = 32,
) -> dict:
    """
    Train PatchCore for a single MVTec category.

    Returns a result dict with keys:
        category, status, training_images, training_time_s, error, output_dir
    """
    if layers is None:
        layers = ["layer2", "layer3"]

    log = setup_logging(category)

    log.info("=" * 60)
    log.info("PatchCore Training  |  Category: %s", category.upper())
    log.info("=" * 60)

    result = {
        "category":       category,
        "status":         "FAILED",
        "training_images": 0,
        "training_time_s": 0.0,
        "error":          "",
        "output_dir":     str(MODELS_DIR / category),
    }

    start_time = time.time()

    try:
        # ── Environment ───────────────────────────────────────
        accelerator, devices = get_accelerator()
        if torch.cuda.is_available():
            log.info("GPU: %s (%.1f GB VRAM)",
                     torch.cuda.get_device_name(0),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)
        else:
            log.info("Using CPU (no CUDA detected)")

        # ── Dataset verification ──────────────────────────────
        _, n_train = verify_dataset(category, log)
        result["training_images"] = n_train

        # ── Anomalib imports ──────────────────────────────────
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        # ── DataModule ────────────────────────────────────────
        log.info("Building DataModule...")
        datamodule = MVTecAD(
            root=str(DATASET_ROOT),
            category=category,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,   # Required on Windows
        )

        datamodule.setup()

        n_train_actual = len(datamodule.train_data)
        n_test         = len(datamodule.test_data)
        n_val          = len(datamodule.val_data)

        assert n_train_actual > 0, (
            f"FATAL: train_data empty for '{category}'. "
            "Check dataset path and anomalib/Pandas compatibility."
        )

        log.info("train_data: %d | test_data: %d | val_data: %d",
                 n_train_actual, n_test, n_val)
        result["training_images"] = n_train_actual

        # ── Model ─────────────────────────────────────────────
        log.info("Building PatchCore model...")
        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
        )
        log.info("Backbone: %s | Layers: %s | Coreset: %.1f%% | Neighbors: %d",
                 backbone, layers, coreset_ratio * 100, num_neighbors)

        # ── Output directory ──────────────────────────────────
        output_dir = MODELS_DIR / category
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Engine + Training ─────────────────────────────────
        log.info("Starting training...")
        engine = Engine(
            accelerator=accelerator,
            devices=devices,
            default_root_dir=str(output_dir),
        )

        engine.fit(model=model, datamodule=datamodule)
        log.info("Training complete.")

        # ── Evaluation ────────────────────────────────────────
        log.info("Running evaluation on test set...")
        test_results = engine.test(model=model, datamodule=datamodule)

        metrics = {}
        if test_results:
            metrics = {k: float(v) for k, v in test_results[0].items()}
            log.info("Test metrics:")
            for k, v in metrics.items():
                log.info("  %-40s: %.4f", k, v)

        # ── Save artifacts ────────────────────────────────────
        config = {
            "category":              category,
            "backbone":              backbone,
            "layers":                layers,
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors":         num_neighbors,
            "batch_size":            batch_size,
        }

        save_model_artifacts(model, engine, output_dir, config, log)

        # metrics.json
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("metrics.json saved : %s", metrics_path)

        # ── Checkpoint location ───────────────────────────────
        checkpoints = sorted(output_dir.rglob("*.ckpt"))
        if checkpoints:
            log.info("Checkpoint         : %s", checkpoints[-1])
        else:
            log.warning("No .ckpt file found in %s", output_dir)

        # ── Timing ───────────────────────────────────────────
        elapsed = time.time() - start_time
        result["training_time_s"] = round(elapsed, 1)

        # ── metadata.json ─────────────────────────────────────
        metadata = {
            "category":              category,
            "backbone":              backbone,
            "layers":                layers,
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors":         num_neighbors,
            "batch_size":            batch_size,
            "training_images":       n_train_actual,
            "training_time_s":       round(elapsed, 1),
            "accelerator":           accelerator,
            "anomalib_version":      _get_anomalib_version(),
            "torch_version":         torch.__version__,
            "metrics":               metrics,
            "status":                "SUCCESS",
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        log.info("metadata.json saved: %s", metadata_path)

        result["status"] = "SUCCESS"
        log.info("=" * 60)
        log.info("SUCCESS  |  %s  |  %.1f s", category.upper(), elapsed)
        log.info("=" * 60)

    except Exception as e:
        elapsed = time.time() - start_time
        result["training_time_s"] = round(elapsed, 1)
        result["error"] = str(e)
        log.error("FAILED   |  %s  |  %s", category.upper(), e, exc_info=True)

    return result


def _get_anomalib_version() -> str:
    try:
        import anomalib
        return anomalib.__version__
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Train PatchCore anomaly detection for one MVTec AD category.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--category", "-c",
        required=True,
        help="MVTec category name (e.g. bottle, leather, cable)",
    )
    parser.add_argument(
        "--backbone",
        default="wide_resnet50_2",
        help="Timm backbone name for feature extraction",
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
        help="Coreset sampling ratio (0.01 = 1%%, recommended for RTX GPUs)",
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
    args = parser.parse_args()

    result = train_category(
        category=args.category,
        backbone=args.backbone,
        layers=args.layers,
        coreset_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
    )

    print()
    if result["status"] == "SUCCESS":
        print(f"[SUCCESS] {args.category}  |  "
              f"{result['training_images']} images  |  "
              f"{result['training_time_s']:.1f}s  |  "
              f"Saved to: {result['output_dir']}")
        sys.exit(0)
    else:
        print(f"[FAILED]  {args.category}  |  Error: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
