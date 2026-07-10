"""
Model Registry
==============
Singleton registry to manage and cache trained PatchCore models.
Compatible with Python 3.12 and Anomalib 2.5.
"""

import logging
from pathlib import Path
from typing import Dict, List

import torch
from anomalib.models import Patchcore

# ============================================================
# Final Thresholds (Benchmark Results)
# ============================================================

MODEL_THRESHOLDS = {
    "bottle": 40.4697,
    "cable": 45.4637,
    "capsule": 28.0012,
    "carpet": 34.8693,
    "grid": 34.4564,
    "hazelnut": 50.6877,
    "leather": 39.7085,
    "metal_nut": 44.0953,
    "pill": 35.4441,
    "screw": 34.8573,
    "tile": 39.5625,
    "toothbrush": 37.3722,
    "transistor": 40.5637,
    "wood": 44.3572,
    "zipper": 31.7607,
}

print("=" * 60)
print("MODEL_REGISTRY FILE LOADED")
print("=" * 60)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry that loads and caches PatchCore models.
    """

    _instance = None
    _models: Dict[str, Patchcore] = {}

    _base_dir = Path("models/patchcore")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True

            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    )
                )
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

    # ============================================================
    # Category Utilities
    # ============================================================

    def list_available(self) -> List[str]:
        """
        Return all trained categories.
        """

        if not self._base_dir.exists():
            return []

        categories = []

        for folder in self._base_dir.iterdir():
            if folder.is_dir():
                categories.append(folder.name)

        return sorted(categories)

    def is_trained(self, category: str) -> bool:
        """
        Check whether a trained checkpoint exists.
        """

        checkpoint = (
            self._base_dir
            / category
            / "Patchcore"
            / "MVTecAD"
            / category
            / "v0"
            / "weights"
            / "lightning"
            / "model.ckpt"
        )

        return checkpoint.exists()

    # ============================================================
    # Threshold Utilities
    # ============================================================

    def get_all_thresholds(self) -> dict:
        """
        Return benchmark thresholds.
        """
        return MODEL_THRESHOLDS.copy()

    def get_threshold(self, category: str) -> float:
        """
        Return threshold for a category.

        Uses benchmark threshold first.
        Falls back to checkpoint threshold.
        """

        if category in MODEL_THRESHOLDS:
            return MODEL_THRESHOLDS[category]

        model = self.load(category)

        threshold = model.post_processor.image_threshold

        if isinstance(threshold, torch.Tensor):
            return float(threshold.item())

        return float(threshold)

    # ============================================================
    # Model Loading
    # ============================================================

    def load(self, category: str) -> Patchcore:
        """
        Load a trained PatchCore checkpoint.
        """

        print("\n========== USING NEW MODEL REGISTRY ==========\n")

        if category in self._models:
            logger.info(f"Using cached model for '{category}'.")
            return self._models[category]

        checkpoint_path = (
            self._base_dir
            / category
            / "Patchcore"
            / "MVTecAD"
            / category
            / "v0"
            / "weights"
            / "lightning"
            / "model.ckpt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n{checkpoint_path}"
            )

        logger.info(f"Loading checkpoint:\n{checkpoint_path}")

        try:
            model = Patchcore.load_from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                map_location="cpu",
            )

            model.eval()

            self._models[category] = model

            logger.info(f"Successfully loaded '{category}'.")

            return model

        except Exception as e:
            logger.exception("Failed to load checkpoint.")

            raise RuntimeError(
                f"Failed to load checkpoint for '{category}'.\n{e}"
            )