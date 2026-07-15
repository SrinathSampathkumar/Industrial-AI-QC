import time
from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/msme_defect_detection-3/weights/best.onnx"
IMAGE_DIR = "test_images/leather"

model = YOLO(MODEL_PATH)

images = sorted(Path(IMAGE_DIR).glob("*.png"))

if len(images) == 0:
    raise FileNotFoundError("No images found.")

print("=" * 60)
print("ONNX CPU BENCHMARK")
print("=" * 60)

times = []

# -------------------------
# Warm-up (NOT timed)
# -------------------------
print("Running warm-up...")

model.predict(
    source=str(images[0]),
    device="cpu",
    verbose=False
)

print("Warm-up completed.\n")

# -------------------------
# Actual Benchmark
# -------------------------
for img in images:

    start = time.perf_counter()

    model.predict(
        source=str(img),
        device="cpu",
        verbose=False
    )

    end = time.perf_counter()

    t = end - start

    times.append(t)

    print(f"{img.name:<20} {t:.4f} sec")
    t = end - start

    times.append(t)

    print(f"{img.name:<20} {t:.4f} sec")

avg = sum(times) / len(times)

print("=" * 60)
print(f"Average CPU inference : {avg:.4f} sec/image")
print(f"Images tested         : {len(images)}")

if avg < 1.5:
    print("PASS ✓ Target achieved (<1.5 sec/image)")
else:
    print("FAIL ✗ Target not achieved")