"""
PatchCore Inference Script
===========================
Runs anomaly detection on new images using a trained PatchCore model.

Usage:
    python scripts/inference_patchcore.py --input path/to/image_or_folder
    python scripts/inference_patchcore.py --input datasets/leather/test/cut/000.png
    python scripts/inference_patchcore.py --input datasets/leather/test/

Outputs:
    - Anomaly score (0 = normal, higher = more anomalous)
    - Anomaly map saved as a heatmap PNG
    - Verdict: NORMAL / ANOMALOUS
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np

# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        )
    ],
)
log = logging.getLogger("inference_patchcore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR  = PROJECT_ROOT / "results" / "patchcore" / "leather"
INFERENCE_OUT = PROJECT_ROOT / "results" / "patchcore" / "leather" / "inference_output"

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def load_model(results_dir: Path, device: torch.device):
    """Load trained PatchCore model from saved checkpoint."""
    from anomalib.models import Patchcore

    # Find checkpoint
    ckpt_files = sorted(results_dir.rglob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No .ckpt found in {results_dir}. Run train_patchcore.py first.")

    ckpt_path = ckpt_files[-1]
    log.info("Loading checkpoint: %s", ckpt_path)

    model = Patchcore.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(device)
    return model


def run_inference_anomalib(image_paths: list[Path], results_dir: Path, output_dir: Path, device: torch.device):
    """Run inference using the Anomalib Engine (recommended path)."""
    from anomalib.engine import Engine
    from anomalib.data.predict import PredictDataset
    from torch.utils.data import DataLoader

    model = load_model(results_dir, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        default_root_dir=str(output_dir),
    )

    results_summary = []

    for image_path in image_paths:
        log.info("-" * 50)
        log.info("Image: %s", image_path.name)

        predict_dataset = PredictDataset(path=str(image_path))
        predict_loader  = DataLoader(predict_dataset, batch_size=1, num_workers=0)

        predictions = engine.predict(
            model=model,
            dataloaders=predict_loader,
        )

        if predictions:
            for pred in predictions:
                score = float(pred.pred_score[0]) if pred.pred_score is not None else None
                label = int(pred.pred_label[0])   if pred.pred_label is not None else None
                verdict = "ANOMALOUS" if label == 1 else "NORMAL"
                log.info("  Score  : %.4f", score if score is not None else -1)
                log.info("  Verdict: %s", verdict)
                results_summary.append({
                    "image": str(image_path),
                    "score": score,
                    "label": label,
                    "verdict": verdict,
                })

    # Save summary
    summary_path = output_dir / "inference_results.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log.info("=" * 50)
    log.info("Results saved: %s", summary_path)
    return results_summary


def run_inference_direct(image_paths: list[Path], results_dir: Path, output_dir: Path, device: torch.device):
    """
    Direct inference using saved model.pt state dict.
    Used as fallback if load_from_checkpoint fails.
    """
    import cv2
    from torchvision import transforms
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    model_pt = results_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"model.pt not found at {model_pt}")

    checkpoint = torch.load(str(model_pt), map_location=device)
    log.info("Loaded model.pt  backbone: %s", checkpoint["backbone"])

    torch_model = PatchcoreModel(
        backbone=checkpoint["backbone"],
        layers=checkpoint["layers"],
        num_neighbors=checkpoint["num_neighbors"],
    )
    torch_model.memory_bank = checkpoint["memory_bank"].to(device)
    torch_model.eval()
    torch_model.to(device)

    # Load threshold
    threshold = 0.5  # default
    threshold_path = results_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            td = json.load(f)
        if isinstance(td.get("image_threshold"), float):
            threshold = td["image_threshold"]
            log.info("Threshold loaded: %.4f", threshold)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Image transforms matching PatchCore defaults
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    results_summary = []

    for image_path in image_paths:
        log.info("-" * 50)
        log.info("Image: %s", image_path.name)

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            log.warning("Could not read: %s", image_path)
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        tensor = transform(img_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = torch_model(tensor)

        score = float(output.pred_score[0])
        verdict = "ANOMALOUS" if score > threshold else "NORMAL"
        log.info("  Score  : %.4f (threshold=%.4f)", score, threshold)
        log.info("  Verdict: %s", verdict)

        # Save anomaly map
        if output.anomaly_map is not None:
            anomaly_map = output.anomaly_map[0, 0].cpu().numpy()
            anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            heatmap = cv2.applyColorMap((anomaly_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # Overlay on original image
            img_resized = cv2.resize(img_bgr, (256, 256))
            overlay = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)
            out_name = image_path.stem + "_anomaly_map.png"
            cv2.imwrite(str(output_dir / out_name), overlay)
            log.info("  Map saved: %s", output_dir / out_name)

        results_summary.append({
            "image": str(image_path),
            "score": score,
            "threshold": threshold,
            "verdict": verdict,
        })

    summary_path = output_dir / "inference_results.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log.info("=" * 50)
    log.info("Results saved: %s", summary_path)
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="PatchCore Leather Anomaly Inference")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an image file or folder containing images",
    )
    parser.add_argument(
        "--output", "-o",
        default=str(INFERENCE_OUT),
        help=f"Output directory for results (default: {INFERENCE_OUT})",
    )
    parser.add_argument(
        "--results-dir", "-r",
        default=str(RESULTS_DIR),
        help=f"Directory containing trained model (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "engine", "direct"],
        default="auto",
        help="Inference mode: engine=use Anomalib Engine, direct=use model.pt, auto=try engine first",
    )
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────
    input_path   = Path(args.input)
    output_dir   = Path(args.output)
    results_dir  = Path(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("PatchCore Inference")
    log.info("=" * 60)
    log.info("Device     : %s", device)
    log.info("Input      : %s", input_path)
    log.info("Output     : %s", output_dir)
    log.info("Model from : %s", results_dir)

    # ── Collect images ───────────────────────────────────────
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.rglob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        )
    else:
        log.error("Input not found: %s", input_path)
        sys.exit(1)

    if not image_paths:
        log.error("No images found in: %s", input_path)
        sys.exit(1)

    log.info("Images     : %d", len(image_paths))

    # ── Run inference ────────────────────────────────────────
    if args.mode == "engine":
        run_inference_anomalib(image_paths, results_dir, output_dir, device)
    elif args.mode == "direct":
        run_inference_direct(image_paths, results_dir, output_dir, device)
    else:  # auto
        try:
            run_inference_anomalib(image_paths, results_dir, output_dir, device)
        except Exception as e:
            log.warning("Engine inference failed (%s), falling back to direct mode.", e)
            run_inference_direct(image_paths, results_dir, output_dir, device)

    log.info("INFERENCE COMPLETE")


if __name__ == "__main__":
    main()
