"""
PatchCore Inference Script
==========================

Run inference on single images using trained PatchCore models.

Features
--------
- ModelRegistry integration
- Category-specific threshold
- Category-specific score normalization (0-100)
- Heatmap generation
- Python 3.12
- Anomalib 2.5
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

project_root = str(Path(__file__).resolve().parent.parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from scripts.registry.model_registry import ModelRegistry
from scripts.utils.scaler import ScoreScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Heatmap Generator
# ---------------------------------------------------------


def generate_heatmap(
    image_bgr: np.ndarray,
    anomaly_map_tensor: torch.Tensor,
    output_path: str,
) -> str:

    anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    if maximum > minimum:
        anomaly_map = (anomaly_map - minimum) / (maximum - minimum)

    heatmap = (anomaly_map * 255).astype(np.uint8)

    h, w = image_bgr.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image_bgr,
        0.6,
        heatmap,
        0.4,
        0,
    )

    cv2.imwrite(output_path, overlay)

    return output_path


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------


def predict_image(
    category: str,
    image_path: str,
    output_dir: str = "reports/heatmaps",
) -> Dict[str, Any]:

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    # -----------------------------------
    # Load model
    # -----------------------------------

    registry = ModelRegistry()

    model = registry.load(category)

    threshold = registry.get_threshold(category)

    scaler = ScoreScaler()

    # -----------------------------------
    # Load image
    # -----------------------------------

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(image_path)

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(image_rgb)

    transform = Compose(
        [
            ToTensor(),
        ]
    )

    image_tensor = transform(
        pil_image
    ).unsqueeze(0)

    # -----------------------------------
    # Run inference
    # -----------------------------------

    logger.info(
        f"Running inference for {category}"
    )

    with torch.no_grad():

        output = model(image_tensor)

    # -----------------------------------
    # Raw Score
    # -----------------------------------

    raw_score = 0.0

    if (
        hasattr(output, "pred_score")
        and output.pred_score is not None
    ):
        raw_score = float(output.pred_score.item())

    # -----------------------------------
    # Normalize Score
    # -----------------------------------

    normalized_score = scaler.normalize(
        raw_score,
        category,
    )

    # -----------------------------------
    # Prediction
    # -----------------------------------

    prediction = "Normal"

    if (
        hasattr(output, "pred_label")
        and output.pred_label is not None
    ):
        if bool(output.pred_label.item()):
            prediction = "Anomaly"

    # -----------------------------------
    # Defect Type
    # -----------------------------------

    if prediction == "Normal":
        defect_type = "good"

    elif image_path.parent.name == "uploads":
        defect_type = "unknown"

    else:
        defect_type = image_path.parent.name

    # -----------------------------------
    # Heatmap
    # -----------------------------------

    heatmap_path = None

    if (
        hasattr(output, "anomaly_map")
        and output.anomaly_map is not None
    ):

        save_dir = (
            Path(output_dir)
            / category
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{category}_{defect_type}_{image_path.name}"
        )

        heatmap_path = generate_heatmap(
            image_bgr=image_bgr,
            anomaly_map_tensor=output.anomaly_map,
            output_path=str(
                save_dir / filename
            ),
        )

    # -----------------------------------
    # Console Log
    # -----------------------------------

    print("\n==============================")
    print("PATCHCORE RESULT")
    print("==============================")
    print("Category          :", category)
    print("Prediction        :", prediction)
    print("Raw Score         :", round(raw_score, 4))
    print(
        "Normalized Score  :",
        round(normalized_score, 2),
        "/100",
    )
    print("Threshold         :", threshold)
    print("==============================\n")

    # -----------------------------------
    # Return
    # -----------------------------------

    return {
        "category": category,
        "defect_type": defect_type,
        "prediction": prediction,
        "raw_score": raw_score,
        "normalized_score": normalized_score,
        "threshold": threshold,
        "heatmap_path": heatmap_path,
    }


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    result = predict_image(
    category="bottle",
    image_path="datasets/bottle/test/good/000.png",
    )

    print("\nReturned Dictionary\n")

    for key, value in result.items():
        print(f"{key:<20}: {value}")