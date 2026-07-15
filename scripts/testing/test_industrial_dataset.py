from ultralytics import YOLO
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

model = YOLO(project_root / "runs/detect/runs/mvtec/yolo11n_mvtec73/weights/best.pt")

root = project_root / "industrial_test"

for folder in root.iterdir():
    if folder.is_dir():
        print(f"\nTesting {folder.name}...")

        model.predict(
            source=str(folder),
            save=True,
            save_txt=True,
            save_conf=True,
            imgsz=640,
            conf=0.25,
            device=0,
            name=folder.name,
            project="runs/industrial_test",
            exist_ok=True
        )

print("\nFinished testing all categories.")