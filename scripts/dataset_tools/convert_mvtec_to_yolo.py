"""Convert MVTec AD defect masks into a YOLO detection dataset.

Only anomalous MVTec test images are converted.  Every connected component in
its ground-truth mask becomes a YOLO bounding box.  Class names are scoped by
category (for example, ``bottle__contamination``), so identically named defect
types from different products are never incorrectly merged.

Examples
--------
    python scripts/dataset_tools/convert_mvtec_to_yolo.py
    python scripts/dataset_tools/convert_mvtec_to_yolo.py --val-ratio 0.25
    python scripts/dataset_tools/convert_mvtec_to_yolo.py --clean

The output directory contains the standard YOLO image/label layout, a
``data.yaml`` file suitable for Ultralytics training, and JSON/CSV conversion
reports.  The script deliberately does not process ``good`` images because
they have no defect masks or detection labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import cv2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is included in requirements.txt
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets_yolo"

MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskSample:
    """A defect image paired with its MVTec ground-truth mask."""

    category: str
    defect_type: str
    image_path: Path
    mask_path: Path

    @property
    def class_name(self) -> str:
        return f"{self.category}__{self.defect_type}"

    @property
    def output_stem(self) -> str:
        # MVTec image basenames repeat between categories and defect folders.
        return f"{self.category}__{self.defect_type}__{self.image_path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MVTec AD ground-truth masks to YOLO bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="MVTec AD dataset root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="YOLO dataset output root")
    parser.add_argument("--categories", nargs="+", choices=MVTEC_CATEGORIES,
                        help="Categories to convert (default: all 15 MVTec categories)")
    parser.add_argument("--val-ratio", type=float, default=0.20,
                        help="Validation fraction, split independently per class")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic split seed")
    parser.add_argument("--min-box-area", type=int, default=1,
                        help="Discard connected components smaller than this pixel area")
    parser.add_argument("--clean", action="store_true",
                        help="Remove existing generated images and labels before conversion")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        default="INFO")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        parser.error("--val-ratio must be greater than 0 and less than 1")
    if args.min_box_area < 1:
        parser.error("--min-box-area must be at least 1")
    return args


def configure_logging(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("mvtec_to_yolo")


def find_image_for_mask(test_dir: Path, mask_path: Path) -> Path | None:
    """Find the image paired with ``000_mask.png`` without assuming its suffix."""
    stem = mask_path.stem
    image_stem = stem[:-5] if stem.endswith("_mask") else stem
    for candidate in sorted(test_dir.glob(f"{image_stem}.*")):
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def discover_samples(source: Path, categories: Iterable[str], log: logging.Logger) -> tuple[list[MaskSample], list[dict]]:
    """Discover only anomalous test images that have an associated mask."""
    samples: list[MaskSample] = []
    errors: list[dict] = []

    for category in categories:
        category_root = source / category
        masks_root = category_root / "ground_truth"
        test_root = category_root / "test"
        if not masks_root.is_dir() or not test_root.is_dir():
            message = "Missing ground_truth or test directory"
            log.error("%s: %s", category, message)
            errors.append({"category": category, "stage": "discovery", "message": message})
            continue

        for defect_dir in sorted(path for path in masks_root.iterdir() if path.is_dir()):
            defect_type = defect_dir.name
            test_dir = test_root / defect_type
            if not test_dir.is_dir():
                message = f"Missing corresponding test directory: {test_dir}"
                log.error("%s/%s: %s", category, defect_type, message)
                errors.append({"category": category, "defect_type": defect_type,
                               "stage": "discovery", "message": message})
                continue

            for mask_path in sorted(path for path in defect_dir.iterdir()
                                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                image_path = find_image_for_mask(test_dir, mask_path)
                if image_path is None:
                    message = f"No test image matches mask {mask_path.name}"
                    log.warning("%s/%s: %s", category, defect_type, message)
                    errors.append({"category": category, "defect_type": defect_type,
                                   "mask": str(mask_path), "stage": "discovery", "message": message})
                    continue
                samples.append(MaskSample(category, defect_type, image_path, mask_path))

    return samples, errors


def split_samples(samples: list[MaskSample], val_ratio: float, seed: int) -> dict[str, list[MaskSample]]:
    """Create a deterministic, class-stratified train/validation split."""
    groups: dict[str, list[MaskSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.class_name].append(sample)

    rng = random.Random(seed)
    splits: dict[str, list[MaskSample]] = {"train": [], "val": []}
    for class_name in sorted(groups):
        group = sorted(groups[class_name], key=lambda item: str(item.image_path))
        rng.shuffle(group)
        # Keep at least one training sample when a class has multiple images.
        val_count = 0 if len(group) == 1 else max(1, min(len(group) - 1, round(len(group) * val_ratio)))
        splits["val"].extend(group[:val_count])
        splits["train"].extend(group[val_count:])
    return splits


def mask_to_yolo_boxes(mask_path: Path, image_width: int, image_height: int,
                       min_box_area: int) -> list[tuple[float, float, float, float]]:
    """Convert every non-trivial connected mask component into normalized xywh."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Mask could not be read")
    if mask.shape[:2] != (image_height, image_width):
        raise ValueError(
            f"Mask dimensions {mask.shape[1]}x{mask.shape[0]} do not match "
            f"image dimensions {image_width}x{image_height}"
        )

    binary_mask = (mask > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < min_box_area:
            continue
        centre_x = (x + width / 2) / image_width
        centre_y = (y + height / 2) / image_height
        boxes.append((centre_x, centre_y, width / image_width, height / image_height))
    return boxes


def prepare_output(output: Path, clean: bool) -> None:
    """Create the required layout and prevent accidental mixing with old data."""
    generated_dirs = (output / "images", output / "labels")
    if clean:
        for directory in generated_dirs:
            if directory.exists():
                shutil.rmtree(directory)
    elif any(directory.exists() and any(directory.rglob("*")) for directory in generated_dirs):
        raise FileExistsError(
            f"{output} already contains images or labels. Use --clean to rebuild it safely."
        )

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(output: Path, class_names: list[str]) -> None:
    """Write a minimal Ultralytics-compatible dataset definition."""
    lines = [
        "# Generated by scripts/dataset_tools/convert_mvtec_to_yolo.py",
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {index}: {json.dumps(name)}" for index, name in enumerate(class_names))
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_sample(sample: MaskSample, split: str, output: Path, class_id: int,
                   min_box_area: int) -> int:
    """Copy one image, write its YOLO label file, and return its box count."""
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image could not be read")
    height, width = image.shape[:2]
    boxes = mask_to_yolo_boxes(sample.mask_path, width, height, min_box_area)
    if not boxes:
        raise ValueError("Mask contains no components after minimum-area filtering")

    image_output = output / "images" / split / f"{sample.output_stem}{sample.image_path.suffix.lower()}"
    label_output = output / "labels" / split / f"{sample.output_stem}.txt"
    shutil.copy2(sample.image_path, image_output)
    label_output.write_text(
        "".join(
            f"{class_id} {centre_x:.6f} {centre_y:.6f} {box_width:.6f} {box_height:.6f}\n"
            for centre_x, centre_y, box_width, box_height in boxes
        ),
        encoding="utf-8",
    )
    return len(boxes)


def write_summary(output: Path, report: dict, class_names: list[str]) -> None:
    """Persist machine-readable JSON and an at-a-glance per-class CSV report."""
    (output / "conversion_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output / "conversion_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("class_id", "class_name", "train_images",
                                                   "val_images", "train_boxes", "val_boxes"))
        writer.writeheader()
        for class_id, class_name in enumerate(class_names):
            counts = report["per_class"][class_name]
            writer.writerow({"class_id": class_id, "class_name": class_name, **counts})


def main() -> int:
    args = parse_args()
    log = configure_logging(args.log_level)
    source = args.source.resolve()
    output = args.output.resolve()
    categories = tuple(args.categories) if args.categories else MVTEC_CATEGORIES

    if not source.is_dir():
        log.error("Source dataset directory does not exist: %s", source)
        return 2

    log.info("Discovering anomalous MVTec samples from %s", source)
    samples, errors = discover_samples(source, categories, log)
    if not samples:
        log.error("No valid image/mask pairs were discovered; no dataset was written.")
        return 1

    class_names = sorted({sample.class_name for sample in samples})
    class_ids = {name: index for index, name in enumerate(class_names)}
    splits = split_samples(samples, args.val_ratio, args.seed)

    try:
        prepare_output(output, args.clean)
    except OSError as error:
        log.error("%s", error)
        return 2

    per_class = {
        name: {"train_images": 0, "val_images": 0, "train_boxes": 0, "val_boxes": 0}
        for name in class_names
    }
    split_images: Counter[str] = Counter()
    split_boxes: Counter[str] = Counter()
    progress_items = [(split, sample) for split in ("train", "val") for sample in splits[split]]
    iterator = tqdm(progress_items, desc="Converting masks", unit="image") if tqdm else progress_items

    for split, sample in iterator:
        try:
            box_count = convert_sample(sample, split, output, class_ids[sample.class_name],
                                       args.min_box_area)
            split_images[split] += 1
            split_boxes[split] += box_count
            per_class[sample.class_name][f"{split}_images"] += 1
            per_class[sample.class_name][f"{split}_boxes"] += box_count
        except Exception as error:  # Keep a long conversion running after bad source files.
            log.exception("Failed to convert %s", sample.image_path)
            errors.append({"category": sample.category, "defect_type": sample.defect_type,
                           "image": str(sample.image_path), "mask": str(sample.mask_path),
                           "stage": "conversion", "message": str(error)})

    write_data_yaml(output, class_names)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "output": str(output),
        "categories_requested": list(categories),
        "class_count": len(class_names),
        "class_names": {str(index): name for index, name in enumerate(class_names)},
        "split": {"seed": args.seed, "validation_ratio": args.val_ratio,
                  "strategy": "deterministic per-class stratified split"},
        "images": {"train": split_images["train"], "val": split_images["val"],
                   "total": split_images["train"] + split_images["val"]},
        "boxes": {"train": split_boxes["train"], "val": split_boxes["val"],
                  "total": split_boxes["train"] + split_boxes["val"]},
        "errors": errors,
        "per_class": per_class,
    }
    write_summary(output, report, class_names)

    log.info("Conversion complete: %d images, %d boxes, %d classes",
             report["images"]["total"], report["boxes"]["total"], len(class_names))
    log.info("Train=%d, Val=%d | Reports: %s",
             report["images"]["train"], report["images"]["val"], output / "conversion_summary.json")
    if errors:
        log.warning("Completed with %d skipped/error samples; see conversion_summary.json", len(errors))
    return 0 if report["images"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
