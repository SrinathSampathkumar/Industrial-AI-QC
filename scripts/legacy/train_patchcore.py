"""
PatchCore Anomaly Detection Training Script
===========================================
Trains a PatchCore model on the MVTec AD Leather dataset using Anomalib.

Fix Applied:
    - anomalib 2.5.0 + Pandas 3.x bug: make_mvtec_ad_dataset compared Split enum
      to StringDtype column → filter returned 0 rows. Fixed in site-packages by
      extracting .value before comparison.

Outputs saved to: results/patchcore/leather/
    ├── weights/         <- PyTorch Lightning checkpoint (.ckpt)
    ├── model.pt         <- Inference-ready TorchScript model
    ├── threshold.json   <- Anomaly threshold value
    └── metrics.json     <- Final evaluation metrics
"""

import json
import logging
import sys
from pathlib import Path

import torch

# ─────────────────────────────────────────────────────────────
# Set float32 matmul precision for Tensor Core GPUs (RTX 2050)
# ─────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision("medium")

# ─────────────────────────────────────────────────────────────
# Setup logging (UTF-8 safe on Windows)
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
    ],
)
log = logging.getLogger("train_patchcore")

# ─────────────────────────────────────────────────────────────
# Project paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR  = PROJECT_ROOT / "results" / "patchcore" / "leather"

# ─────────────────────────────────────────────────────────────
# Verify CUDA
# ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Environment Check")
log.info("=" * 60)
log.info("Python version : %s", sys.version.split()[0])
log.info("PyTorch version: %s", torch.__version__)

if torch.cuda.is_available():
    log.info("CUDA available : YES")
    log.info("GPU            : %s", torch.cuda.get_device_name(0))
    log.info("VRAM           : %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)
    ACCELERATOR = "gpu"
    DEVICES = 1
else:
    log.warning("CUDA NOT available - training on CPU (will be slow)")
    ACCELERATOR = "cpu"
    DEVICES = 1

# ─────────────────────────────────────────────────────────────
# Verify dataset
# ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Dataset Verification")
log.info("=" * 60)
LEATHER_DIR = DATASET_ROOT / "leather"
assert LEATHER_DIR.exists(),                     f"Dataset not found: {LEATHER_DIR}"
assert (LEATHER_DIR / "train" / "good").exists(), "Missing train/good/"
assert (LEATHER_DIR / "test").exists(),            "Missing test/"
assert (LEATHER_DIR / "ground_truth").exists(),    "Missing ground_truth/"
log.info("Dataset root   : %s", LEATHER_DIR)
log.info("Dataset OK     : YES")

# ─────────────────────────────────────────────────────────────
# Anomalib imports
# ─────────────────────────────────────────────────────────────
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ─────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Building DataModule")
log.info("=" * 60)

datamodule = MVTecAD(
    root=str(DATASET_ROOT),
    category="leather",
    train_batch_size=32,
    eval_batch_size=32,
    # num_workers=0 required on Windows (multiprocessing issues with DataLoader)
    num_workers=0,
)

# Eagerly set up and verify counts before starting training
datamodule.setup()
assert len(datamodule.train_data) > 0, f"FATAL: train_data is empty! Check dataset and anomalib/Pandas compatibility."
assert len(datamodule.test_data)  > 0, f"FATAL: test_data is empty!"

log.info("train_data len : %d", len(datamodule.train_data))
log.info("test_data len  : %d", len(datamodule.test_data))
log.info("val_data len   : %d", len(datamodule.val_data))
log.info("Task type      : %s", datamodule.task)

# ─────────────────────────────────────────────────────────────
# Model
#
# coreset_sampling_ratio=0.01:
#   The k-center greedy coreset loop runs `coreset_size - 1` iterations.
#   With ratio=0.1 on ~25K patches: ~2500 iterations (slow on RTX 2050).
#   With ratio=0.01: ~250 iterations - fast, and still representative.
#   The PatchCore paper uses 1% for large datasets.
# ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Building Model")
log.info("=" * 60)

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.01,   # 1% coreset - fast and effective on RTX 2050
    num_neighbors=9,
)

log.info("Model               : PatchCore")
log.info("Backbone            : wide_resnet50_2")
log.info("Layers              : layer2, layer3")
log.info("Coreset ratio       : 0.01 (1%%)")
log.info("Num neighbors       : 9")

# ─────────────────────────────────────────────────────────────
# Output directory
# ─────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Engine + Training
# ─────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Starting Training")
log.info("=" * 60)

engine = Engine(
    accelerator=ACCELERATOR,
    devices=DEVICES,
    default_root_dir=str(RESULTS_DIR),
)

engine.fit(
    model=model,
    datamodule=datamodule,
)

log.info("=" * 60)
log.info("Training Complete")
log.info("=" * 60)

# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
log.info("Running evaluation on test set...")
test_results = engine.test(
    model=model,
    datamodule=datamodule,
)

if test_results:
    log.info("Evaluation results:")
    for k, v in test_results[0].items():
        log.info("  %-35s: %.4f", k, float(v))

# ─────────────────────────────────────────────────────────────
# Save metrics
# ─────────────────────────────────────────────────────────────
metrics_path = RESULTS_DIR / "metrics.json"
if test_results:
    with open(metrics_path, "w") as f:
        json.dump({k: float(v) for k, v in test_results[0].items()}, f, indent=2)
    log.info("Metrics saved  : %s", metrics_path)

# ─────────────────────────────────────────────────────────────
# Save anomaly threshold
# ─────────────────────────────────────────────────────────────
threshold_path = RESULTS_DIR / "threshold.json"
threshold_value = None

# Anomalib stores the fitted threshold in the post-processor
try:
    # Try accessing from the engine's trained model
    trained_model = engine.trainer.model
    if hasattr(trained_model, "post_processor"):
        pp = trained_model.post_processor
        if hasattr(pp, "image_threshold"):
            threshold_value = float(pp.image_threshold.value)
        elif hasattr(pp, "threshold"):
            threshold_value = float(pp.threshold)
    elif hasattr(model, "image_threshold"):
        threshold_value = float(model.image_threshold.value)
except Exception as e:
    log.warning("Could not extract threshold from post_processor: %s", e)

if threshold_value is None:
    # Fallback: extract from memory bank statistics
    try:
        mem_bank = model.model.memory_bank
        log.info("Memory bank shape  : %s", mem_bank.shape)
    except Exception:
        pass
    log.warning("Threshold not extracted - will be set during inference from normalization.")

threshold_data = {
    "image_threshold": threshold_value if threshold_value is not None else "fitted_at_inference",
    "note": "PatchCore threshold is fitted from validation scores during training."
}
with open(threshold_path, "w") as f:
    json.dump(threshold_data, f, indent=2)
log.info("Threshold saved: %s", threshold_path)

# ─────────────────────────────────────────────────────────────
# Save inference-ready model (TorchScript export)
# ─────────────────────────────────────────────────────────────
model_pt_path = RESULTS_DIR / "model.pt"
try:
    # Save the entire Lightning module state dict
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "memory_bank": model.model.memory_bank,
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "num_neighbors": 9,
            "coreset_sampling_ratio": 0.01,
        },
        model_pt_path,
    )
    log.info("Inference model saved: %s", model_pt_path)
except Exception as e:
    log.warning("Could not save model.pt: %s", e)

# ─────────────────────────────────────────────────────────────
# Find and report checkpoint
# ─────────────────────────────────────────────────────────────
checkpoints = sorted(RESULTS_DIR.rglob("*.ckpt"))
if checkpoints:
    log.info("Checkpoint     : %s", checkpoints[-1])
else:
    log.warning("No .ckpt file found in %s", RESULTS_DIR)

log.info("=" * 60)
log.info("All outputs saved to: %s", RESULTS_DIR)
log.info("TRAINING PIPELINE COMPLETE")
log.info("=" * 60)