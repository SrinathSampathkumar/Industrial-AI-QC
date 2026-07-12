from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from scripts.inference.yolo_detector import YOLODetector

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE = PROJECT_ROOT / "datasets/bottle/test/broken_large/000.png"

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid"
]

detector = YOLODetector()

print("="*80)
print("YOLO CONFIG COMPARISON")
print("="*80)

for category in CATEGORIES:

    result = detector.detect(str(IMAGE), category)

    print(f"\nCategory        : {category}")
    print(f"Processing Time : {result['processing_time']} sec")
    print(f"Detections      : {result['detection_count']}")

    if result["detections"]:
        print("Confidence Scores:")
        for d in result["detections"]:
            print(f"  {d['confidence']:.4f}")
    else:
        print("No detections")