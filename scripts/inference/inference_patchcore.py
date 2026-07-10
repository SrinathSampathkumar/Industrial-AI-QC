"""
PatchCore Inference Script
==========================
Run inference on single images using the trained PatchCore models
from the ModelRegistry.

Generates anomaly heatmaps overlaid on the original images.
Compatible with Python 3.12 and Anomalib 2.5.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

# Ensure scripts module is accessible when running from root
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from scripts.registry.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def generate_heatmap(image_bgr: np.ndarray, anomaly_map_tensor: torch.Tensor, output_path: str) -> str:
    """
    Generate an anomaly heatmap overlay and save it to disk.
    
    Args:
        image_bgr: The original BGR image loaded by OpenCV.
        anomaly_map_tensor: The 2D or 3D tensor output from anomalib's inference.
        output_path: Where to save the resulting heatmap image.
        
    Returns:
        The absolute path to the saved heatmap.
    """
    # Move tensor to CPU and convert to numpy array
    anomaly_map = anomaly_map_tensor.squeeze().cpu().numpy()
    
    # Normalize anomaly_map to [0-255] range
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    if max_val > min_val:
        normalized_map = (anomaly_map - min_val) / (max_val - min_val)
    else:
        normalized_map = anomaly_map
        
    heatmap_gray = (normalized_map * 255).astype(np.uint8)
    
    # Resize heatmap to match the original image resolution
    h, w = image_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap_gray, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Apply JET colormap for visualization (red = high anomaly)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay heatmap onto original image (60% original, 40% heatmap)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Save to disk
    cv2.imwrite(output_path, overlay)
    return output_path


def predict_image(category: str, image_path: str, output_dir: str = "reports/heatmaps") -> Dict[str, Any]:
    """
    Run PatchCore inference on a single test image.
    
    Args:
        category: MVTec AD category (e.g., 'bottle', 'zipper').
        image_path: Path to the image file to test.
        output_dir: Directory where the heatmap will be saved.
        
    Returns:
        Dictionary containing prediction results, anomaly score, threshold, and heatmap path.
    """
    img_path_obj = Path(image_path)
    if not img_path_obj.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # 1. Access Registry (Singleton)
    registry = ModelRegistry()
    
    # 2. Load cached model and fetch threshold
    model = registry.load(category)
    threshold = registry.get_threshold(category)
    
    # 3. Read image with OpenCV (BGR format)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Failed to decode image at {image_path}")
        
    # Convert BGR -> RGB for preprocessing
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for torchvision transforms
    pil_img = Image.fromarray(img_rgb)
    
    # Convert image to tensor only.
    # Resize and normalization are handled automatically by the
    # PatchCore model's built-in PreProcessor.
    transform = Compose([
    ToTensor()
    ]
    )
    
    # Add batch dimension: [1, 3, 256, 256]
    img_tensor = transform(pil_img).unsqueeze(0)
    
    # 4. Run Inference Native Pass
    logger.info(f"Running inference for '{category}' on image '{img_path_obj.name}'...")
    with torch.no_grad():
        output = model(img_tensor)
        
    # Extract prediction data from InferenceBatch
    score = output.pred_score.item() if hasattr(output, "pred_score") and output.pred_score is not None else "N/A"
    
    is_anomaly = False
    if hasattr(output, "pred_label") and output.pred_label is not None:
        is_anomaly = bool(output.pred_label.item())
        
    prediction_label = "Anomaly" if is_anomaly else "Normal"
    
    # 5. Generate Heatmap
    heatmap_saved_path = None
    if hasattr(output, "anomaly_map") and output.anomaly_map is not None:
        out_dir_path = Path(output_dir) / category
        out_dir_path.mkdir(parents=True, exist_ok=True)

        if prediction_label == "Normal":
            defect_type = "good"
        elif img_path_obj.parent.name == "uploads":
            defect_type = "unknown"
        else:
            defect_type = img_path_obj.parent.name
        
        heatmap_filename = f"{category}_{defect_type}_{img_path_obj.name}"
        heatmap_target = str(out_dir_path / heatmap_filename)
        
        heatmap_saved_path = generate_heatmap(img_bgr, output.anomaly_map, heatmap_target)
        print(heatmap_saved_path)
        
        logger.info(f"Heatmap generated at: {heatmap_saved_path}")

        
        
    return {
        "category": category,
        "defect_type": defect_type,
        "prediction": prediction_label,
        "anomaly_score": score,
        "threshold": threshold,
        "heatmap_path": heatmap_saved_path
    }


if __name__ == "__main__":
    # Standalone execution for quick verification
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    test_category = "bottle"
    test_img = "datasets/bottle/test/broken_large/000.png"
    
    try:
        results = predict_image(test_category, test_img)
        print("\n--- INFERENCE RESULTS ---")
        for key, value in results.items():
            print(f"{key.capitalize():<15}: {value}")
        print("-------------------------")
    except Exception as e:
        logger.error(f"Inference test failed: {e}")
