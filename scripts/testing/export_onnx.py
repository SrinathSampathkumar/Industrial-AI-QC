from ultralytics import YOLO

model = YOLO("runs/detect/runs/msme_defect_detection-3/weights/best.pt")

model.export(
    format="onnx",
    opset=12,
    dynamic=True,
    simplify=True
)

print("ONNX export completed.")