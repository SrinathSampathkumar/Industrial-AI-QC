"""
YOLO Detector
=============

Loads category-specific YOLO configurations and performs
object detection using Ultralytics YOLO.

Includes:
- Category-specific confidence threshold
- Category-specific IOU threshold
- Minimum defect area filtering
- Detection logging

Author: Srinath
"""

from pathlib import Path
import time
import yaml
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO detector with category-specific configuration loading.
    """

    def __init__(self):
        """
        Initialize YOLO detector.
        """

        self.project_root = Path(__file__).resolve().parents[2]

        self.config_dir = self.project_root / "yolo_configs"

        self.model = YOLO(
            self.project_root / "yolo11n.pt"
        )

    def load_config(self, category: str) -> dict:
        """
        Load YAML configuration for a category.
        """

        config_path = self.config_dir / f"{category}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"No configuration found for '{category}'."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def detect(self, image_path: str, category: str) -> dict:
        """
        Run YOLO inference using category-specific configuration.
        """

        config = self.load_config(category)

        expected_classes = config["expected_classes"]

        min_area = config["min_defect_size"]

        start_time = time.time()

        results = self.model.predict(
            source=image_path,
            conf=config["confidence_threshold"],
            iou=config["iou_threshold"],
            imgsz=config["input_size"],
            verbose=False,
        )

        processing_time = time.time() - start_time

        result = results[0]

        detections = []

        filtered_small_boxes = 0

        if result.boxes is not None:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                class_name = result.names[cls_id]

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                width = x2 - x1
                height = y2 - y1

                area = width * height

                bbox = [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ]

                # Keep only expected classes
                if class_name not in expected_classes:
                    continue

                # Ignore tiny detections
                if area < min_area:
                    filtered_small_boxes += 1
                    continue

                detections.append(
                    {
                        "class": class_name,
                        "confidence": round(confidence, 4),
                        "area": round(area, 2),
                        "bbox": bbox,
                    }
                )

        output = {
            "category": category,
            "image": image_path,
            "processing_time": round(processing_time, 4),
            "min_area": min_area,
            "detection_count": len(detections),
            "filtered_small_boxes": filtered_small_boxes,
            "detections": detections,
        }

        return output


if __name__ == "__main__":

    detector = YOLODetector()

    output = detector.detect(
        image_path="datasets/bottle/test/good/000.png",
        category="bottle",
    )

    print("=" * 60)
    print("YOLO DETECTION RESULTS")
    print("=" * 60)

    print(f"Category              : {output['category']}")
    print(f"Image                 : {output['image']}")
    print(f"Processing Time       : {output['processing_time']} sec")
    print(f"Minimum Area Filter   : {output['min_area']}")
    print(f"Detection Count       : {output['detection_count']}")
    print(f"Filtered Small Boxes  : {output['filtered_small_boxes']}")

    print("\nDetections")
    print("=" * 60)

    if output["detections"]:

        for detection in output["detections"]:

            print(
                f"Class={detection['class']} | "
                f"Confidence={detection['confidence']} | "
                f"Area={detection['area']} | "
                f"BBox={detection['bbox']}"
            )

    else:

        print("No detections.")

    print("=" * 60)