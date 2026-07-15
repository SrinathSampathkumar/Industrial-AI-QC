from pathlib import Path
import shutil
import random
import yaml

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC = PROJECT_ROOT / "datasets_yolo"
DST = PROJECT_ROOT / "datasets_yolo_v2"

WEAK = {
    "grid",
    "toothbrush",
    "transistor",
    "screw",
    "wood",
    "pill",
    "capsule",
    "cable",
}

STRONG = {
    "bottle",
    "hazelnut",
    "leather",
    "metal_nut",
    "carpet",
    "tile",
    "zipper",
}


def copy_subset(split):
    src_img = SRC / "images" / split
    src_lbl = SRC / "labels" / split

    dst_img = DST / "images" / split
    dst_lbl = DST / "labels" / split

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    images = list(src_img.glob("*.png"))

    for img in images:
        name = img.stem

        category = name.split("__")[0]

        keep = False

        if category in WEAK:
            keep = True

        elif category in STRONG:
            keep = random.random() < 0.25

        if not keep:
            continue

        shutil.copy2(img, dst_img / img.name)

        label = src_lbl / (name + ".txt")

        if label.exists():
            shutil.copy2(label, dst_lbl / label.name)


copy_subset("train")
copy_subset("val")

with open(SRC / "data.yaml") as f:
    data = yaml.safe_load(f)

data["path"] = "."

with open(DST / "data.yaml", "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)

print("Improvement dataset created successfully.")