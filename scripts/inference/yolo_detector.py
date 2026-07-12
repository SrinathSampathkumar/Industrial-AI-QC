"""
YOLO Detector
=============

Loads category-specific YOLO configurations and performs
object detection using Ultralytics YOLO.

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

        expected_classes = config["expected_classes"]

        if result.boxes is not None:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                class_name = result.names[cls_id]

                confidence = float(box.conf[0])

                bbox = [
                    round(float(x), 2)
                    for x in box.xyxy[0].tolist()
                ]

                # Keep only expected classes
                if class_name not in expected_classes:
                    continue

                detections.append(
                    {
                        "class": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": bbox,
                    }
                )

        output = {
            "category": category,
            "image": image_path,
            "processing_time": round(processing_time, 4),
            "detection_count": len(detections),
            "detections": detections,
        }

        return output


if __name__ == "__main__":

    detector = YOLODetector()

    output = detector.detect(
        image_path="datasets/bottle/test/broken_large/000.png",
        category="bottle",
    )

    print("=" * 60)
    print("YOLO DETECTION RESULTS")
    print("=" * 60)

    print(f"Category          : {output['category']}")
    print(f"Image             : {output['image']}")
    print(f"Processing Time   : {output['processing_time']} sec")
    print(f"Detection Count   : {output['detection_count']}")

    print("\nDetections")

    for detection in output["detections"]:
        print(
            f"Class={detection['class']} | "
            f"Confidence={detection['confidence']} | "
            f"BBox={detection['bbox']}"
        )

    print("=" * 60)