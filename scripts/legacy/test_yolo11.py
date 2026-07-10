from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


def main():
    # ==========================================
    # Configuration
    # ==========================================
    MODEL_PATH = "yolo11n.pt"
    INPUT_DIR = Path("test_images/leather")
    OUTPUT_DIR = Path("outputs/yolo11_mtvec")
    CONFIDENCE = 0.25

    # ==========================================
    # Select Device
    # ==========================================
    if torch.cuda.is_available():
        device = 0
        print(f"[INFO] GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] CUDA not available. Using CPU.")

    # ==========================================
    # Load Model
    # ==========================================
    print(f"[INFO] Loading {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # ==========================================
    # Create Output Folder
    # ==========================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # Find Images
    # ==========================================
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
        "*.tif",
        "*.tiff",
        "*.webp",
    ]

    image_files = []

    for ext in image_extensions:
        image_files.extend(INPUT_DIR.glob(ext))
        image_files.extend(INPUT_DIR.glob(ext.upper()))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"[ERROR] No images found in:\n{INPUT_DIR.resolve()}")
        return

    print(f"[INFO] Found {len(image_files)} images.\n")

    # ==========================================
    # Run Inference
    # ==========================================
    for image_path in image_files:

        print("=" * 50)
        print(f"Image : {image_path.name}")

        try:

            results = model.predict(
                source=str(image_path),
                conf=CONFIDENCE,
                device=device,
                save=False,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()

            save_path = OUTPUT_DIR / image_path.name

            cv2.imwrite(str(save_path), annotated)

            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                print("No detection")

            else:
                for box in boxes:
                    cls = int(box.cls[0])
                    name = result.names[cls]
                    conf = float(box.conf[0])

                    print(f"Class      : {name}")
                    print(f"Confidence : {conf:.4f}")

            print(f"Saved      : {save_path}")

        except Exception as e:
            print(f"[ERROR] {e}")

        print("=" * 50)
        print()

    print("==============================================")
    print("Inference Finished Successfully")
    print(f"Results saved in:\n{OUTPUT_DIR.resolve()}")
    print("==============================================")


if __name__ == "__main__":
    main()