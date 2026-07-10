import sys
import time
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from anomalib.models import Patchcore

def main():
    models_dir = Path("models/patchcore")
    if not models_dir.exists():
        print(f"Error: Directory {models_dir} not found.")
        sys.exit(1)
        
    categories = sorted([d.name for d in models_dir.iterdir() if d.is_dir()])
    if not categories:
        print("No categories found in models/patchcore/")
        sys.exit(1)
        
    print(f"Discovered {len(categories)} categories: {', '.join(categories)}\n")
    
    # Standard Anomalib transforms
    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    passed = 0
    failed = 0
    start_time = time.time()
    
    for category in categories:
        cat_start = time.time()
        print("-" * 50)
        print(f"Testing Category: {category.upper()}")
        print("-" * 50)
        
        try:
            model_dir = models_dir / category
            model_path = model_dir / "model.pt"
            threshold_path = model_dir / "threshold.json"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            # Load threshold
            threshold_val = "UNKNOWN"
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    t_data = json.load(f)
                    threshold_val = t_data.get("image_threshold", "UNKNOWN")
                    
            # Load model
            data = torch.load(model_path, weights_only=False)
            model = Patchcore(
                backbone=data.get("backbone", "wide_resnet50_2"),
                layers=data.get("layers", ["layer2", "layer3"]),
                coreset_sampling_ratio=data.get("coreset_sampling_ratio", 0.01),
                num_neighbors=data.get("num_neighbors", 9),
            )
            model.load_state_dict(data["model_state_dict"])
            model.eval()
            
            # Load test image
            test_good_dir = Path(f"datasets/{category}/test/good")
            image_paths = list(test_good_dir.glob("*.png"))
            if not image_paths:
                raise FileNotFoundError(f"No test images found in {test_good_dir}")
                
            img = Image.open(image_paths[0]).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                output = model(img_tensor)
                
            anomaly_score = output.pred_score.item() if hasattr(output, "pred_score") and output.pred_score is not None else "N/A"
            if hasattr(output, "pred_label") and output.pred_label is not None:
                prediction_str = "Anomaly" if output.pred_label.item() else "Normal"
            else:
                prediction_str = "Unknown"
                
            print(f"Category   : {category}")
            print(f"Prediction : {prediction_str}")
            print(f"Score      : {anomaly_score}")
            print(f"Threshold  : {threshold_val}")
            print(f"PASS (Took {time.time() - cat_start:.2f}s)")
            passed += 1
            
        except Exception as e:
            print(f"Category   : {category}")
            print(f"FAIL: {str(e)}")
            failed += 1
            
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Total Categories : {len(categories)}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"Execution Time   : {total_time:.2f} seconds")
    print("=" * 50)

if __name__ == "__main__":
    main()
