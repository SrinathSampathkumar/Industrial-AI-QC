from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    model.train(
        data="roboflow_dataset/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,      # Windows-friendly
        cache="disk",   # Better than RAM
        project="runs",
        name="msme_defect_detection"
    )


if __name__ == "__main__":
    main()