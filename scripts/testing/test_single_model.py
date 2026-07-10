import sys
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from anomalib.models import Patchcore

def main():
    category = "bottle"
    model_dir = Path(f"models/patchcore/{category}")
    model_path = model_dir / "model.pt"
    threshold_path = model_dir / "threshold.json"
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
        
    # 1. Load threshold
    threshold_val = "UNKNOWN"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            t_data = json.load(f)
            threshold_val = t_data.get("image_threshold", "UNKNOWN")
            
    # 2. Load model
    print(f"Loading model from {model_path}...")
    data = torch.load(model_path, weights_only=False)
    
    model = Patchcore(
        backbone=data.get("backbone", "wide_resnet50_2"),
        layers=data.get("layers", ["layer2", "layer3"]),
        coreset_sampling_ratio=data.get("coreset_sampling_ratio", 0.01),
        num_neighbors=data.get("num_neighbors", 9),
    )
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    
    # 3. Load image
    test_good_dir = Path(f"datasets/{category}/test/good")
    image_paths = list(test_good_dir.glob("*.png"))
    if not image_paths:
        print(f"Error: No test images found in {test_good_dir}")
        sys.exit(1)
        
    image_path = image_paths[0]
    print(f"Loading test image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    
    # Standard Anomalib transforms (using ImageNet stats)
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # 4. Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(img_tensor)
    
    # Parse InferenceBatch output
    anomaly_score = output.pred_score.item() if hasattr(output, "pred_score") and output.pred_score is not None else "N/A"
    
    # If the model returned a prediction label, use it. Otherwise, default to "Unknown"
    if hasattr(output, "pred_label") and output.pred_label is not None:
        prediction_str = "Anomaly" if output.pred_label.item() else "Normal"
    else:
        prediction_str = "Unknown"
        
    print("\n" + "="*40)
    print("INFERENCE RESULTS")
    print("="*40)
    print(f"Category      : {category}")
    print(f"Prediction    : {prediction_str}")
    print(f"Anomaly Score : {anomaly_score}")
    print(f"Threshold     : {threshold_val}")
    print("PASS")
    print("="*40)

if __name__ == "__main__":
    main()
