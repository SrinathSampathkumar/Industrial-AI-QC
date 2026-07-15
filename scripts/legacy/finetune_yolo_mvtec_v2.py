from ultralytics import YOLO


def main():
    model = YOLO("models/checkpoints/best_v1.pt")

    model.train(
        data="datasets_yolo/data.yaml",
        epochs=20,
        imgsz=800,
        batch=12,
        lr0=0.001,
        cache="disk",
        workers=4,
        device=0,
        project="runs/mvtec",
        name="yolo11n_mvtec73_v2",
        exist_ok=True,
        pretrained=False,
    )


if __name__ == "__main__":
    main()