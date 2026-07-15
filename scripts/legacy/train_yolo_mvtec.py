from ultralytics import YOLO
import torch
torch.cuda.empty_cache()


def main():

    model = YOLO("yolo11n.pt")

    model.train(
        data="datasets_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        cache="ram",
        project="runs/mvtec",
        name="yolo11n_mvtec73",
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()