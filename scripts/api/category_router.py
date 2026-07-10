"""
Category Router
===============

Routes an inspection request to the correct trained PatchCore model.
"""

from typing import Dict, Any
from pathlib import Path

from scripts.registry.model_registry import ModelRegistry
from scripts.inference.inference_patchcore import predict_image


registry = ModelRegistry()


def route_inspection(image_path: str, category_key: str) -> Dict[str, Any]:
    """
    Route an inspection request to the appropriate model.

    Args:
        image_path: Path to the input image.
        category_key: Dataset category (bottle, cable, carpet, etc.)

    Returns:
        Prediction dictionary.
    """

    category_key = category_key.lower().strip()

    # Check if the category has a trained model
    if not registry.is_trained(category_key):
        return {
            "status": "model_not_ready",
            "category": category_key,
            "message": f"No trained model available for '{category_key}'."
        }

    # Check image exists
    if not Path(image_path).exists():
        return {
            "status": "error",
            "message": f"Image not found: {image_path}"
        }

    # Run inference
    result = predict_image(
        category=category_key,
        image_path=image_path
    )

    result["status"] = "success"

    return result