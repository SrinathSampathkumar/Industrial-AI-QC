"""
Model Registry
==============
Singleton registry to manage and cache trained PatchCore models.
Compatible with Python 3.12 and Anomalib 2.5.
"""

import logging
from pathlib import Path
from typing import Dict, List
import json
from unicodedata import category
from xml.parsers.expat import model

import torch
from anomalib.models import Patchcore

# ============================================================
# Final Thresholds (Benchmark Results)
# ============================================================

THRESHOLD_FILE = (
    Path("models")
    / "thresholds"
    / "thresholds.json"
)

if THRESHOLD_FILE.exists():
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        THRESHOLD_DATA = json.load(f)
else:
    THRESHOLD_DATA = {}

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
        Check whether at least one trained checkpoint exists.
        Automatically finds the latest version.
        """

        checkpoint_root = (
            self._base_dir
            / category
            / "Patchcore"
            / "MVTecAD"
            / category
        )

        if not checkpoint_root.exists():
            return False

        versions = sorted(
            checkpoint_root.glob("v*"),
            key=lambda p: int(p.name[1:])
        )

        if not versions:
            return False

        checkpoint = (
            versions[-1]
            / "weights"
            / "lightning"
            / "model.ckpt"
        )

        return checkpoint.exists()

    def get_all_thresholds(self):

        return {
            category: data["threshold"]
            for category, data in THRESHOLD_DATA.items()
        }

    def get_threshold(self, category: str) -> float:

        if category in THRESHOLD_DATA:
            return float(
                THRESHOLD_DATA[category]["threshold"]
            )

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

        checkpoint_root = (
            self._base_dir
            / category
            / "Patchcore"
            / "MVTecAD"
            / category
        )

        if not checkpoint_root.exists():
            raise FileNotFoundError(
                f"No checkpoint directory found for '{category}'."
            )

        versions = sorted(
            checkpoint_root.glob("v*"),
            key=lambda p: int(p.name[1:])
        )

        if not versions:
            raise FileNotFoundError(
                f"No trained versions found for '{category}'."
            )

        latest = versions[-1]

        checkpoint_path = (
            latest
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