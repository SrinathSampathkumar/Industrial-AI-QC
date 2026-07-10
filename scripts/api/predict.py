"""
Prediction API Wrapper
======================
Provides a clean, JSON-compatible interface to run PatchCore inference.
Designed for easy integration with web frameworks (like FastAPI).
Compatible with Python 3.12 and Anomalib 2.5.
"""

import logging
from pathlib import Path
from typing import Any, Dict

# Ensure scripts module is accessible when running from root
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.registry.model_registry import ModelRegistry
from scripts.api.category_router import route_inspection

logger = logging.getLogger(__name__)


def run_prediction(category: str, image_path: str) -> Dict[str, Any]:
    """
    Validates inputs and runs inference, returning a JSON-compatible dictionary.
    
    Args:
        category: The MVTec AD category (e.g., 'bottle').
        image_path: Path to the image file to be analyzed.
        
    Returns:
        JSON-compatible dictionary with prediction results or error message.
    """
    try:
        # 1. Validate Image
        img_path_obj = Path(image_path)
        if not img_path_obj.exists():
            return {
                "status": "error",
                "message": f"Image file not found: {image_path}"
            }
        if not img_path_obj.is_file():
            return {
                "status": "error",
                "message": f"Path is not a file: {image_path}"
            }
            
        # 2. Validate Category & Model
        registry = ModelRegistry()
        if not registry.is_trained(category):
            return {
                "status": "error",
                "message": f"Model for category '{category}' is missing or not fully trained."
            }
            
        
        
        # 3. Call core inference logic
        logger.info(f"Initiating prediction for '{category}' on '{img_path_obj.name}'...")
        result = route_inspection(
            image_path=str(img_path_obj),
            category_key=category
            )
        
        # 4. Construct Success Response
        return {
            "status": "success",
            "category": result.get("category"),
            "defect_type": result.get("defect_type"),
            "prediction": result.get("prediction"),
            "anomaly_score": round(float(result.get("anomaly_score")), 4),
            "heatmap_path": result.get("heatmap_path")
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="PatchCore Prediction API")

    parser.add_argument(
        "--category",
        required=True,
        help="Category name (e.g. bottle, cable, zipper)"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    args = parser.parse_args()

    response = run_prediction(args.category, args.image)

    print("\n--- JSON OUTPUT ---")
    print(json.dumps(response, indent=4))
    print("-------------------")