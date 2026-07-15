"""
create_dataset_from_masks_v2.py
================================
Production-quality YOLO dataset generation pipeline for the Industrial-AI-QC project.

Supports three dataset layouts, automatically detected:

  FORMAT A  (test_images/)
      category/bad/  -> PNG image + BMP mask (same stem)
      category/good/ -> defect-free PNG images

  FORMAT B  (datasets/ - MVTec AD standard layout)
      category/train/good/            -> defect-free train images
      category/test/{defect}/         -> anomalous test images
      category/ground_truth/{defect}/ -> binary mask files (*_mask.png)

  FORMAT C  (nested MVTec, e.g. datasets/screw/screw/)
      Same as Format B but data lives one level deeper.

Output: datasets_yolo_masks/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml  validation_report.txt  conversion_report.json

Usage:
    python scripts/dataset_tools/create_dataset_from_masks_v2.py
    python scripts/dataset_tools/create_dataset_from_masks_v2.py --clean
    python scripts/dataset_tools/create_dataset_from_masks_v2.py --val-ratio 0.25
    python scripts/dataset_tools/create_dataset_from_masks_v2.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_MVTEC_SOURCE: Path = PROJECT_ROOT / "datasets"
DEFAULT_CUSTOM_SOURCE: Path = PROJECT_ROOT / "test_images"
DEFAULT_OUTPUT: Path = PROJECT_ROOT / "datasets_yolo_masks"

IMAGE_EXTENSIONS: frozenset = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"})

# Minimum mask contour area in pixels squared - smaller blobs are noise
MIN_CONTOUR_AREA: int = 25

RANDOM_SEED: int = 42
DEFAULT_VAL_RATIO: float = 0.20


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One image/label pair destined for the YOLO dataset."""

    category: str
    defect_type: str           # "good" for defect-free samples
    image_path: Path
    mask_path: Optional[Path]  # None for good/background samples

    @property
    def class_name(self) -> str:
        """Scoped class name: category__defect (e.g. bottle__broken_large)."""
        return f"{self.category}__{self.defect_type}"

    @property
    def is_good(self) -> bool:
        return self.defect_type == "good"

    @property
    def output_stem(self) -> str:
        """
        Collision-safe output filename stem.
        Encodes category, defect type, and original image stem so that
        identical filenames across different categories never collide.
        """
        return f"{self.category}__{self.defect_type}__{self.image_path.stem}"


@dataclass
class ConversionStats:
    """Mutable counters updated throughout the pipeline."""

    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0
    good_images: int = 0
    defect_images: int = 0
    skipped_images: int = 0
    warnings: list = field(default_factory=list)
    per_category: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("dataset_v2")


LOG = configure_logging()


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production YOLO dataset generator for Industrial-AI-QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-mvtec", type=Path, default=DEFAULT_MVTEC_SOURCE,
        help="MVTec-style root directory (Format B / Format C).",
    )
    parser.add_argument(
        "--source-custom", type=Path, default=DEFAULT_CUSTOM_SOURCE,
        help="Custom bad/good directory root (Format A).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="YOLO dataset output root.",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
        help="Fraction of each class assigned to validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for the deterministic train/val split.",
    )
    parser.add_argument(
        "--min-contour-area", type=int, default=MIN_CONTOUR_AREA,
        help="Discard mask contours smaller than this area (pixels squared).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing images/ and labels/ before generating.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Verbosity level.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be strictly between 0 and 1.")
    if args.min_contour_area < 1:
        parser.error("--min-contour-area must be >= 1.")
    return args


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _is_image_file(path: Path) -> bool:
    """Return True when the file extension is a recognised image format."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _detect_mvtec_root(candidate: Path) -> Optional[Path]:
    """
    Determine whether `candidate` is a direct or one-level nested MVTec root.

    Returns the Path that contains train/ + test/ + ground_truth/, or None.
    """
    # Format B: direct layout
    if (candidate / "test").is_dir() and (candidate / "ground_truth").is_dir():
        return candidate

    # Format C: nested layout (e.g. datasets/screw/screw/)
    nested = candidate / candidate.name
    if (
        nested.is_dir()
        and (nested / "test").is_dir()
        and (nested / "ground_truth").is_dir()
    ):
        return nested

    return None


# ---------------------------------------------------------------------------
# FORMAT A discovery  (test_images/)
# ---------------------------------------------------------------------------

def _defect_from_stem(stem: str) -> str:
    """
    Parse the defect type from a Format A image stem.

    Pattern: {split}_{defect}_{number}
    Examples:
        test_broken_large_000  -> broken_large
        test_contamination_001 -> contamination
        train_good_000         -> good
    """
    for prefix in ("test_", "train_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break

    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]

    return stem  # fallback


def discover_format_a(source: Path, stats: ConversionStats) -> list:
    """
    Discover all samples from a Format A directory tree.

    source/<category>/bad/  - PNG + BMP mask pairs
    source/<category>/good/ - defect-free PNG images

    Returns a flat list of Sample objects.
    """
    samples = []

    if not source.is_dir():
        LOG.warning("Format A source not found: %s -- skipping.", source)
        return samples

    for cat_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        category = cat_dir.name
        cat_samples = []

        bad_dir = cat_dir / "bad"
        if bad_dir.is_dir():
            for img in sorted(bad_dir.glob("*.png")):
                mask = img.with_suffix(".bmp")
                if not mask.exists():
                    msg = f"[{category}] Missing BMP mask for {img.name} -- skipped."
                    LOG.warning(msg)
                    stats.warnings.append(msg)
                    stats.skipped_images += 1
                    continue

                defect = _defect_from_stem(img.stem)
                if defect == "good":
                    cat_samples.append(Sample(category, "good", img, None))
                else:
                    cat_samples.append(Sample(category, defect, img, mask))

        good_dir = cat_dir / "good"
        if good_dir.is_dir():
            for img in sorted(
                p for p in good_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            ):
                cat_samples.append(Sample(category, "good", img, None))

        if not cat_samples:
            msg = f"[{category}] Format A -- no samples found."
            LOG.warning(msg)
            stats.warnings.append(msg)
        else:
            LOG.debug("[%s] Format A -- %d samples.", category, len(cat_samples))

        samples.extend(cat_samples)

    return samples


# ---------------------------------------------------------------------------
# FORMAT B / C discovery  (datasets/)
# ---------------------------------------------------------------------------

def discover_format_bc(source: Path, stats: ConversionStats) -> list:
    """
    Discover samples from an MVTec-style (Format B or C) directory tree.

    Returns a flat list of Sample objects.
    """
    samples = []

    if not source.is_dir():
        LOG.warning("Format B/C source not found: %s -- skipping.", source)
        return samples

    for cat_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        category = cat_dir.name
        root = _detect_mvtec_root(cat_dir)

        if root is None:
            LOG.debug("[%s] No MVTec layout detected -- skipping.", category)
            continue

        fmt_label = "Format C" if root != cat_dir else "Format B"
        cat_samples = []

        # Good images from train/good/
        train_good = root / "train" / "good"
        if train_good.is_dir():
            for img in sorted(p for p in train_good.iterdir() if _is_image_file(p)):
                cat_samples.append(Sample(category, "good", img, None))

        # Good images from test/good/
        test_good = root / "test" / "good"
        if test_good.is_dir():
            for img in sorted(p for p in test_good.iterdir() if _is_image_file(p)):
                cat_samples.append(Sample(category, "good", img, None))

        # Defect images paired with ground-truth masks
        test_root = root / "test"
        gt_root = root / "ground_truth"

        if not test_root.is_dir():
            msg = f"[{category}] Missing test/ directory."
            LOG.warning(msg)
            stats.warnings.append(msg)
        elif not gt_root.is_dir():
            msg = f"[{category}] Missing ground_truth/ directory."
            LOG.warning(msg)
            stats.warnings.append(msg)
        else:
            for defect_dir in sorted(d for d in test_root.iterdir() if d.is_dir()):
                defect = defect_dir.name
                if defect == "good":
                    continue

                gt_dir = gt_root / defect
                if not gt_dir.is_dir():
                    msg = f"[{category}/{defect}] No ground_truth subdir -- skipped."
                    LOG.warning(msg)
                    stats.warnings.append(msg)
                    continue

                for img in sorted(p for p in defect_dir.iterdir() if _is_image_file(p)):
                    mask = gt_dir / f"{img.stem}_mask.png"

                    if not mask.exists():
                        candidates = sorted(gt_dir.glob(f"{img.stem}*"))
                        if candidates:
                            mask = candidates[0]
                        else:
                            msg = (
                                f"[{category}/{defect}] No mask for "
                                f"{img.name} -- skipped."
                            )
                            LOG.warning(msg)
                            stats.warnings.append(msg)
                            stats.skipped_images += 1
                            continue

                    cat_samples.append(Sample(category, defect, img, mask))

        if not cat_samples:
            msg = f"[{category}] {fmt_label} -- no samples found."
            LOG.warning(msg)
            stats.warnings.append(msg)
        else:
            LOG.debug(
                "[%s] %s -- %d samples.", category, fmt_label, len(cat_samples)
            )

        samples.extend(cat_samples)

    return samples


# ---------------------------------------------------------------------------
# Mask to YOLO bounding boxes
# ---------------------------------------------------------------------------

def mask_to_yolo_boxes(
    mask_path: Path,
    img_w: int,
    img_h: int,
    min_area: int,
) -> list:
    """
    Convert a binary mask to normalised YOLO bounding boxes.

    Parameters
    ----------
    mask_path : Path to the mask file (grayscale or BGR).
    img_w     : Width  of the corresponding image in pixels.
    img_h     : Height of the corresponding image in pixels.
    min_area  : Contours with pixel area < min_area are discarded.

    Returns
    -------
    List of (x_center, y_center, width, height) each in [0.0, 1.0].

    Raises
    ------
    ValueError if the mask cannot be read.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Some BMP files stored as BGR are all-zero after grayscale read
    if binary.max() == 0:
        bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        xc = max(0.0, min(1.0, (x + bw / 2.0) / img_w))
        yc = max(0.0, min(1.0, (y + bh / 2.0) / img_h))
        nw = max(0.0, min(1.0, bw / img_w))
        nh = max(0.0, min(1.0, bh / img_h))

        if nw > 0.0 and nh > 0.0:
            boxes.append((xc, yc, nw, nh))

    return boxes


# ---------------------------------------------------------------------------
# Stratified train/val split
# ---------------------------------------------------------------------------

def stratified_split(samples: list, val_ratio: float, seed: int) -> tuple:
    """
    Reproducible, class-stratified train/validation split.

    For every unique class (category__defect), shuffle with `seed` then assign
    round(len * val_ratio) to val, clamped to at least 1 when the class has
    more than one sample.

    Returns (train_samples, val_samples).
    """
    rng = random.Random(seed)
    groups: dict = defaultdict(list)

    for s in samples:
        groups[s.class_name].append(s)

    train, val = [], []

    for cls in sorted(groups):
        group = sorted(groups[cls], key=lambda s: str(s.image_path))
        rng.shuffle(group)

        if len(group) <= 1:
            val_n = 0
        else:
            val_n = max(1, min(len(group) - 1, round(len(group) * val_ratio)))

        val.extend(group[:val_n])
        train.extend(group[val_n:])

    return train, val


# ---------------------------------------------------------------------------
# Output directory management
# ---------------------------------------------------------------------------

def prepare_output(output: Path, clean: bool) -> dict:
    """
    Create the standard YOLO directory layout under `output`.

    If `clean` is True, any existing images/ and labels/ are removed first.

    Returns a mapping of logical names to their Paths:
        img_train, img_val, lbl_train, lbl_val
    """
    dirs = {
        "img_train": output / "images" / "train",
        "img_val":   output / "images" / "val",
        "lbl_train": output / "labels" / "train",
        "lbl_val":   output / "labels" / "val",
    }

    if clean:
        for parent in (output / "images", output / "labels"):
            if parent.exists():
                shutil.rmtree(parent)
                LOG.info("Removed: %s", parent)

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


# ---------------------------------------------------------------------------
# Sample writing
# ---------------------------------------------------------------------------

def write_sample(
    sample: Sample,
    split: str,
    dirs: dict,
    class_id: int,
    min_area: int,
    stats: ConversionStats,
) -> bool:
    """
    Copy the image and write the YOLO label for one sample.

    Good images receive an empty label file (background class).
    Returns True on success, False when the sample is skipped due to an error.
    """
    img_dir = dirs[f"img_{split}"]
    lbl_dir = dirs[f"lbl_{split}"]
    stem = sample.output_stem
    img_dst = img_dir / f"{stem}{sample.image_path.suffix.lower()}"
    lbl_dst = lbl_dir / f"{stem}.txt"

    # Copy image
    try:
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Missing: {sample.image_path}")
        shutil.copy2(sample.image_path, img_dst)
    except Exception as exc:
        msg = f"[{sample.category}] Cannot copy {sample.image_path.name}: {exc}"
        LOG.warning(msg)
        stats.warnings.append(msg)
        stats.skipped_images += 1
        return False

    # Write label
    try:
        if sample.is_good or sample.mask_path is None:
            # Empty label file = YOLO negative sample (background only)
            lbl_dst.write_text("", encoding="utf-8")
        else:
            img = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"cv2 cannot decode: {sample.image_path}")

            img_h, img_w = img.shape[:2]
            boxes = mask_to_yolo_boxes(sample.mask_path, img_w, img_h, min_area)

            if not boxes:
                msg = (
                    f"[{sample.category}/{sample.defect_type}] "
                    f"Mask {sample.mask_path.name} yielded no boxes -- "
                    f"empty label written."
                )
                LOG.warning(msg)
                stats.warnings.append(msg)
                lbl_dst.write_text("", encoding="utf-8")
            else:
                lines = [
                    f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    for xc, yc, bw, bh in boxes
                ]
                lbl_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

    except Exception as exc:
        msg = (
            f"[{sample.category}] Cannot write label for "
            f"{sample.image_path.name}: {exc}"
        )
        LOG.warning(msg)
        stats.warnings.append(msg)
        img_dst.unlink(missing_ok=True)
        stats.skipped_images += 1
        return False

    if split == "train":
        stats.train_images += 1
        stats.train_labels += 1
    else:
        stats.val_images += 1
        stats.val_labels += 1

    if sample.is_good:
        stats.good_images += 1
    else:
        stats.defect_images += 1

    return True


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------

def write_data_yaml(output: Path, class_names: list) -> None:
    """Emit an Ultralytics-compatible data.yaml descriptor."""
    lines = [
        "# Generated by scripts/dataset_tools/create_dataset_from_masks_v2.py",
        f"# {datetime.now(timezone.utc).isoformat()}",
        f"path: {output.as_posix()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(class_names)}",
        "",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")

    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("data.yaml written -- %d classes.", len(class_names))


# ---------------------------------------------------------------------------
# Post-generation validation
# ---------------------------------------------------------------------------

def validate_output(output: Path) -> list:
    """
    Integrity checks on the generated YOLO dataset.

    Verifies per split (train + val):
      1. Every image has a label file.
      2. Every label has an image file.
      3. Every non-empty label line has exactly 5 fields.
      4. All bounding-box coordinates lie within [0, 1].

    Returns a list of error strings (empty = all clear).
    """
    errors = []

    for split in ("train", "val"):
        img_dir = output / "images" / split
        lbl_dir = output / "labels" / split

        if not img_dir.is_dir() or not lbl_dir.is_dir():
            errors.append(f"[{split}] images/ or labels/ directory missing.")
            continue

        img_stems = {p.stem for p in img_dir.iterdir() if p.is_file()}
        lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.is_file()}

        for stem in sorted(img_stems - lbl_stems):
            errors.append(f"[{split}] Image without label: {stem}")
        for stem in sorted(lbl_stems - img_stems):
            errors.append(f"[{split}] Label without image: {stem}")

        for lbl in sorted(lbl_dir.glob("*.txt")):
            text = lbl.read_text(encoding="utf-8").strip()
            if not text:
                continue

            for lineno, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(
                        f"[{split}] {lbl.name}:{lineno} -- "
                        f"expected 5 fields, got {len(parts)}: {line!r}"
                    )
                    continue
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    errors.append(
                        f"[{split}] {lbl.name}:{lineno} -- "
                        f"non-numeric coordinate: {line!r}"
                    )
                    continue
                for v in coords:
                    if not 0.0 <= v <= 1.0:
                        errors.append(
                            f"[{split}] {lbl.name}:{lineno} -- "
                            f"coordinate {v:.6f} outside [0,1]: {line!r}"
                        )

    return errors


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(
    output: Path,
    stats: ConversionStats,
    class_names: list,
    val_errors: list,
    args: argparse.Namespace,
) -> None:
    """Write conversion_report.json and validation_report.txt."""

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "mvtec":  str(args.source_mvtec),
            "custom": str(args.source_custom),
        },
        "output": str(output),
        "settings": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "min_contour_area": args.min_contour_area,
        },
        "summary": {
            "categories":        len(stats.per_category),
            "classes":           len(class_names),
            "train_images":      stats.train_images,
            "val_images":        stats.val_images,
            "train_labels":      stats.train_labels,
            "val_labels":        stats.val_labels,
            "good_images":       stats.good_images,
            "defect_images":     stats.defect_images,
            "skipped_images":    stats.skipped_images,
            "warnings":          len(stats.warnings),
            "validation_errors": len(val_errors),
        },
        "class_names":       {str(i): n for i, n in enumerate(class_names)},
        "per_category":      stats.per_category,
        "warnings":          stats.warnings,
        "validation_errors": val_errors,
    }

    json_path = output / "conversion_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    SEP = "=" * 72
    lines = [
        SEP,
        "  INDUSTRIAL-AI-QC  DATASET VALIDATION REPORT",
        f"  Generated : {datetime.now(timezone.utc).isoformat()}",
        SEP,
    ]

    if val_errors:
        lines.append(f"\n  [FAIL] {len(val_errors)} validation error(s):\n")
        lines.extend(f"    * {e}" for e in val_errors)
    else:
        lines.append("\n  [PASS] All validation checks passed.\n")

    if stats.warnings:
        lines.append(f"\n  [WARN] {len(stats.warnings)} warning(s):\n")
        lines.extend(f"    * {w}" for w in stats.warnings)

    lines += [
        "",
        SEP,
        "  DATASET SUMMARY",
        SEP,
        f"  Number of Categories : {len(stats.per_category)}",
        f"  Number of Classes    : {len(class_names)}",
        f"  Train Images         : {stats.train_images}",
        f"  Validation Images    : {stats.val_images}",
        f"  Train Labels         : {stats.train_labels}",
        f"  Validation Labels    : {stats.val_labels}",
        f"  Good Images          : {stats.good_images}",
        f"  Defect Images        : {stats.defect_images}",
        f"  Skipped Images       : {stats.skipped_images}",
        f"  Warnings             : {len(stats.warnings)}",
        f"  Validation Errors    : {len(val_errors)}",
        "",
        "  CLASS LIST:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"    {i:>3}: {name}")

    txt_path = output / "validation_report.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("Reports written: %s  %s", json_path.name, txt_path.name)


# ---------------------------------------------------------------------------
# Internal pipeline helpers
# ---------------------------------------------------------------------------

def _process_split(
    split_samples: list,
    split: str,
    dirs: dict,
    class_to_id: dict,
    min_area: int,
    stats: ConversionStats,
) -> None:
    """Write all samples for one split, grouped by category for tidy logging."""
    by_cat: dict = defaultdict(list)
    for s in split_samples:
        by_cat[s.category].append(s)

    for category in sorted(by_cat):
        written = 0
        for sample in by_cat[category]:
            cid = class_to_id.get(sample.class_name, -1)
            if write_sample(sample, split, dirs, cid, min_area, stats):
                written += 1
                c = stats.per_category.setdefault(
                    category, {"train": 0, "val": 0, "good": 0, "defect": 0}
                )
                c[split] += 1
                c["good" if sample.is_good else "defect"] += 1

        LOG.debug(
            "[%s/%s] %d / %d written.",
            split, category, written, len(by_cat[category]),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    global LOG
    LOG = configure_logging(args.log_level)

    output = args.output.resolve()
    stats = ConversionStats()

    LOG.info("=" * 60)
    LOG.info("Industrial-AI-QC  YOLO Dataset Generator v2")
    LOG.info("=" * 60)
    LOG.info("MVTec source   : %s", args.source_mvtec)
    LOG.info("Custom source  : %s", args.source_custom)
    LOG.info("Output         : %s", output)
    LOG.info("Val ratio      : %.2f  |  Seed: %d", args.val_ratio, args.seed)
    LOG.info("Min contour px : %d", args.min_contour_area)

    # Step 1 -- Discover samples
    LOG.info("\n[Step 1/7] Discovering samples ...")
    all_samples = []

    fmt_a = discover_format_a(args.source_custom, stats)
    LOG.info("  Format A (test_images): %d samples", len(fmt_a))
    all_samples.extend(fmt_a)

    fmt_bc = discover_format_bc(args.source_mvtec, stats)
    LOG.info("  Format B/C (datasets) : %d samples", len(fmt_bc))
    all_samples.extend(fmt_bc)

    if not all_samples:
        LOG.error("No samples found in any source. Aborting.")
        return 1

    LOG.info("  Total                 : %d samples", len(all_samples))

    # Step 2 -- Class registry
    LOG.info("\n[Step 2/7] Building class registry ...")
    defect_classes = sorted(
        {s.class_name for s in all_samples if not s.is_good}
    )
    class_to_id = {name: i for i, name in enumerate(defect_classes)}
    LOG.info("  Defect classes: %d", len(defect_classes))

    # Step 3 -- Train/val split
    LOG.info(
        "\n[Step 3/7] Splitting (val_ratio=%.2f, seed=%d) ...",
        args.val_ratio, args.seed,
    )
    train_set, val_set = stratified_split(all_samples, args.val_ratio, args.seed)
    LOG.info("  Train: %d  |  Val: %d", len(train_set), len(val_set))

    # Step 4 -- Prepare output
    LOG.info("\n[Step 4/7] Preparing output directories ...")
    dirs = prepare_output(output, args.clean)

    # Step 5 -- Write train
    LOG.info("\n[Step 5/7] Writing train split ...")
    _process_split(train_set, "train", dirs, class_to_id, args.min_contour_area, stats)

    # Step 6 -- Write val
    LOG.info("\n[Step 6/7] Writing val split ...")
    _process_split(val_set, "val", dirs, class_to_id, args.min_contour_area, stats)

    # Per-category table
    LOG.info("\n  Per-Category Summary:")
    LOG.info("  %-22s  %6s  %5s  %6s  %8s", "Category", "train", "val", "good", "defect")
    LOG.info("  " + "-" * 52)
    for cat in sorted(stats.per_category):
        c = stats.per_category[cat]
        LOG.info(
            "  %-22s  %6d  %5d  %6d  %8d",
            cat, c.get("train", 0), c.get("val", 0),
            c.get("good", 0), c.get("defect", 0),
        )

    # data.yaml
    write_data_yaml(output, defect_classes)

    # Step 7 -- Validate
    LOG.info("\n[Step 7/7] Validating output ...")
    val_errors = validate_output(output)
    if val_errors:
        LOG.warning("  %d validation error(s).", len(val_errors))
        for e in val_errors:
            LOG.warning("    %s", e)
    else:
        LOG.info("  Validation PASSED.")

    write_reports(output, stats, defect_classes, val_errors, args)

    # Final summary
    LOG.info("\n" + "=" * 60)
    LOG.info("DATASET GENERATION COMPLETE")
    LOG.info("=" * 60)
    LOG.info("  Number of Categories : %d", len(stats.per_category))
    LOG.info("  Number of Classes    : %d", len(defect_classes))
    LOG.info("  Train Images         : %d", stats.train_images)
    LOG.info("  Validation Images    : %d", stats.val_images)
    LOG.info("  Train Labels         : %d", stats.train_labels)
    LOG.info("  Validation Labels    : %d", stats.val_labels)
    LOG.info("  Good Images          : %d", stats.good_images)
    LOG.info("  Defect Images        : %d", stats.defect_images)
    LOG.info("  Skipped              : %d", stats.skipped_images)
    LOG.info("  Warnings             : %d", len(stats.warnings))
    LOG.info("  Validation Errors    : %d", len(val_errors))
    LOG.info("  Output               : %s", output)
    LOG.info("=" * 60)

    return 0 if not val_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
