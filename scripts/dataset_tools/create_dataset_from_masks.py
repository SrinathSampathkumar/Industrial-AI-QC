from pathlib import Path
import shutil
import random
import cv2

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "test_images"
DST = ROOT / "datasets_yolo_masks"

IMG_TRAIN = DST / "images" / "train"
IMG_VAL = DST / "images" / "val"
LBL_TRAIN = DST / "labels" / "train"
LBL_VAL = DST / "labels" / "val"

for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
    p.mkdir(parents=True, exist_ok=True)

classes = []
class_to_id = {}


def save_labels(img_path, mask_path, classname):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return

    _, th = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = mask.shape

    labels = []

    if classname not in class_to_id:
        class_to_id[classname] = len(classes)
        classes.append(classname)

    cid = class_to_id[classname]

    for c in contours:

        if cv2.contourArea(c) < 5:
            continue

        x, y, bw, bh = cv2.boundingRect(c)

        xc = (x + bw / 2) / w
        yc = (y + bh / 2) / h
        bw /= w
        bh /= h

        labels.append(
            f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    if len(labels) == 0:
        return

    if random.random() < 0.2:
        img_dst = IMG_VAL
        lbl_dst = LBL_VAL
    else:
        img_dst = IMG_TRAIN
        lbl_dst = LBL_TRAIN

    shutil.copy(img_path, img_dst / img_path.name)

    with open(lbl_dst / (img_path.stem + ".txt"), "w") as f:
        f.write("\n".join(labels))


categories = sorted([d for d in SRC.iterdir() if d.is_dir()])

for category in categories:

    print(f"\nProcessing {category.name}")

    root = category

    # screw has screw/screw/
    if (root / category.name).exists():
        root = root / category.name

    # -----------------------------
    # FORMAT 1 : bad + bmp masks
    # -----------------------------
    if (root / "bad").exists():

        bad = root / "bad"

        for img_path in sorted(bad.glob("*.png")):

            mask_path = img_path.with_suffix(".bmp")

            if not mask_path.exists():
                continue

            stem = img_path.stem
            parts = stem.split("_")

            defect = "_".join(parts[1:-1])

            classname = f"{category.name}__{defect}"

            save_labels(img_path, mask_path, classname)

    # -----------------------------
    # FORMAT 2 : MVTec structure
    # -----------------------------
    elif (root / "test").exists() and (root / "ground_truth").exists():

        test_root = root / "test"
        gt_root = root / "ground_truth"

        for defect_dir in sorted(test_root.iterdir()):

            if not defect_dir.is_dir():
                continue

            if defect_dir.name == "good":
                continue

            gt_dir = gt_root / defect_dir.name

            if not gt_dir.exists():
                continue

            for img_path in sorted(defect_dir.glob("*.png")):

                mask_name = img_path.stem + "_mask.png"
                mask_path = gt_dir / mask_name

                if not mask_path.exists():
                    continue

                classname = f"{category.name}__{defect_dir.name}"

                save_labels(img_path, mask_path, classname)

    else:

        print("Skipped (unknown format)")


yaml = DST / "data.yaml"

with open(yaml, "w") as f:

    f.write(
f"""path: {DST.as_posix()}
train: images/train
val: images/val

names:
"""
    )

    for i, c in enumerate(classes):
        f.write(f"  {i}: {c}\n")


print("\nDone.")
print("Classes:", len(classes))
print("Train images:", len(list(IMG_TRAIN.glob("*.png"))))
print("Val images:", len(list(IMG_VAL.glob("*.png"))))