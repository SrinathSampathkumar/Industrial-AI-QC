"""
Hybrid Inspection Pipeline
==========================
Combines PatchCore (Anomaly Detection) and YOLO11 (Defect Localization/Classification).

Flow:
1. PatchCore detects if product is Normal or Anomaly.
2. If Anomaly, YOLO localizes and classifies the specific defect.
3. If YOLO finds nothing (or low confidence), falls back to generic "Anomaly".

Author: Srinath
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

# Add project root to path for direct execution
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Project imports
from scripts.inference.inference_patchcore import predict_image

logger = logging.getLogger(__name__)


class HybridInspectionPipeline:
    """
    Production-ready hybrid pipeline.
    Loads YOLO once in memory to avoid subprocess overhead.
    (PatchCore is currently loaded per-call in predict_image; we will optimize that in Phase 2).
    """

    def __init__(self, yolo_weights_path: str = "models/checkpoints/best_masks_v1.pt"):
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        weights = Path(yolo_weights_path)
        if not weights.is_absolute():
            weights = self.project_root / weights
            
        if not weights.exists():
            logger.warning(f"YOLO weights not found at {weights}. Using base model.")
            weights = self.project_root / "yolo11n.pt"

        logger.info(f"Loading YOLO model from {weights}")
        self.yolo_model = YOLO(str(weights))
        
    def inspect(self, image_path: str, category: str, output_dir: str = "reports/heatmaps") -> Dict[str, Any]:
        """
        Run the full hybrid pipeline on a single image.
        
        Args:
            image_path: Path to the image
            category: MVTec category (e.g., 'bottle')
            output_dir: Where PatchCore saves the GradCAM heatmap
            
        Returns:
            Dictionary containing the final inspection result.
        """
        start_time = time.time()
        
        # ---------------------------------------------------------
        # 1. PatchCore (Anomaly Detection)
        # ---------------------------------------------------------
        logger.info(f"Running PatchCore for {category} on {image_path}")
        try:
            pc_result = predict_image(category=category, image_path=image_path, output_dir=output_dir)
        except Exception as e:
            logger.error(f"PatchCore failed: {e}")
            return {"status": "ERROR", "message": f"PatchCore execution failed: {e}"}

        # ---------------------------------------------------------
        # 2. Decision Logic
        # ---------------------------------------------------------
        if pc_result["prediction"] == "Normal":
            # Product is GOOD
            elapsed = time.time() - start_time
            return {
                "status": "PASS",
                "category": category,
                "defect_classification": "None",
                "confidence_score": 1.0,  # Implicitly 100% confident it's a pass based on PC threshold
                "patchcore": pc_result,
                "yolo": None,
                "processing_time_s": round(elapsed, 3)
            }

        # ---------------------------------------------------------
        # 3. YOLO (Localization & Classification)
        # ---------------------------------------------------------
        logger.info(f"Anomaly detected by PatchCore. Running YOLO for localization.")
        yolo_results = self.yolo_model.predict(
            source=image_path,
            conf=0.20,  # Slightly lower confidence to catch subtle defects (PatchCore already verified it's an anomaly)
            verbose=False
        )[0]
        
        detections = []
        best_defect = "Unknown Anomaly"
        highest_conf = 0.0
        
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_results.names[cls_id]
                conf = float(box.conf[0])
                
                # Verify the prediction belongs to the current category
                # (Dataset classes are formatted like "bottle__broken_large")
                if class_name.startswith(f"{category}__"):
                    defect_name = class_name.split("__")[-1]
                    
                    detections.append({
                        "class": defect_name,
                        "confidence": round(conf, 4),
                        "bbox": box.xyxy[0].tolist()
                    })
                    
                    if conf > highest_conf:
                        highest_conf = conf
                        best_defect = defect_name
                        
        elapsed = time.time() - start_time
        
        # ---------------------------------------------------------
        # 4. Final Output Construction
        # ---------------------------------------------------------
        return {
            "status": "FAIL",
            "category": category,
            "defect_classification": best_defect,
            "confidence_score": round(highest_conf if detections else (pc_result["normalized_score"] / 100.0), 4),
            "patchcore": pc_result,
            "yolo": {
                "detections": detections,
                "highest_confidence": round(highest_conf, 4),
                "detection_count": len(detections)
            },
            "processing_time_s": round(elapsed, 3)
        }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    
    pipeline = HybridInspectionPipeline()
    
    # Test on a known image
    test_image = "datasets/bottle/test/good/000.png"
    if Path(test_image).exists():
        result = pipeline.inspect(image_path=test_image, category="bottle")
        print("\n--- Pipeline Output ---")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image}")
