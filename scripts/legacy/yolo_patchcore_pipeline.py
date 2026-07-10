"""
YOLO + PatchCore Inspection Pipeline
====================================
Combines YOLOv11 for product detection/cropping and PatchCore for anomaly detection.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# Ensure we can import from project root and sibling scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend.gradcam import YOLOGradCAM, overlay_heatmap
from inference_patchcore import run_inference_direct, run_inference_anomalib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout
)
log = logging.getLogger("yolo_patchcore_pipeline")

RESULTS_DIR = PROJECT_ROOT / "results" / "patchcore" / "leather"
PIPELINE_OUT = PROJECT_ROOT / "results" / "pipeline_output"
GRADCAM_OUT  = PROJECT_ROOT / "results" / "gradcam"

def main():
    parser = argparse.ArgumentParser(description="YOLO + PatchCore Pipeline")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--yolo-model", default=str(PROJECT_ROOT / "yolo11n.pt"), help="Path to YOLO model")
    parser.add_argument("--mode", default="direct", choices=["auto", "engine", "direct"], help="Inference mode for PatchCore")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Image not found: {input_path}")
        return

    PIPELINE_OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. YOLO Detection
    log.info("Loading YOLO model...")
    yolo_model = YOLO(args.yolo_model)
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        log.error(f"Could not read image: {input_path}")
        return
        
    results = yolo_model(img_bgr, verbose=False)
    
    detected = False
    confidence = 0.0
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        best_idx = torch.argmax(boxes.conf).item()
        best_box = boxes[best_idx]
        confidence = float(best_box.conf.item())
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        detected = True
        log.info(f"YOLO detected object with confidence {confidence:.4f} at [{x1}, {y1}, {x2}, {y2}]")
        roi_bgr = img_bgr[y1:y2, x1:x2]
    else:
        log.info("YOLO did not detect any product. Using full image as ROI.")
        roi_bgr = img_bgr.copy()
        x1, y1, x2, y2 = 0, 0, img_bgr.shape[1], img_bgr.shape[0]

    # Save ROI
    roi_path = PIPELINE_OUT / f"roi_{input_path.name}"
    cv2.imwrite(str(roi_path), roi_bgr)
    
    # Generate GradCAM for every image (shows backbone attention whether or not YOLO fired)
    gradcam_path_str = None
    GRADCAM_OUT.mkdir(parents=True, exist_ok=True)
    try:
        log.info("Generating GradCAM heatmap...")
        gradcam = YOLOGradCAM(yolo_model)
        cam = gradcam.generate(img_bgr)
        if cam is not None:
            gradcam_img = overlay_heatmap(img_bgr, cam)
            gradcam_path = GRADCAM_OUT / f"{input_path.stem}_gradcam.png"
            cv2.imwrite(str(gradcam_path), gradcam_img)
            gradcam_path_str = str(gradcam_path)
            log.info(f"GradCAM saved to {gradcam_path}")
        else:
            log.warning("GradCAM returned None — no heatmap generated")
    except Exception as e:
        log.error(f"Failed to generate GradCAM: {e}")
    
    # 2. PatchCore Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Running PatchCore inference on ROI... (Mode: {args.mode})")
    
    # Disable logs from inference_patchcore being too noisy? No, it's fine.
    try:
        if args.mode == "engine":
            pc_results = run_inference_anomalib([roi_path], RESULTS_DIR, PIPELINE_OUT, device)
        elif args.mode == "direct":
            pc_results = run_inference_direct([roi_path], RESULTS_DIR, PIPELINE_OUT, device)
        else:
            try:
                pc_results = run_inference_anomalib([roi_path], RESULTS_DIR, PIPELINE_OUT, device)
            except Exception as e:
                log.warning(f"Engine inference failed ({e}), falling back to direct mode.")
                pc_results = run_inference_direct([roi_path], RESULTS_DIR, PIPELINE_OUT, device)
    except Exception as e:
        log.error(f"PatchCore inference failed: {e}")
        return

    if not pc_results:
        log.error("PatchCore inference returned no results.")
        return
        
    pc_result = pc_results[0]
    score = pc_result.get("score", 0.0)
    verdict = pc_result.get("verdict", "UNKNOWN")
    threshold = pc_result.get("threshold", "N/A")
    if threshold == "N/A":
        # Check threshold.json if engine mode didn't provide it
        threshold_path = RESULTS_DIR / "threshold.json"
        if threshold_path.exists():
            with open(threshold_path) as f:
                td = json.load(f)
                if "image_threshold" in td:
                    threshold = td["image_threshold"]
    
    # 3. Save Final Annotated Image
    final_img = img_bgr.copy()
    heatmap_path = PIPELINE_OUT / f"{roi_path.stem}_anomaly_map.png"
    
    if heatmap_path.exists():
        heatmap_img = cv2.imread(str(heatmap_path))
        if heatmap_img is not None:
            # Resize heatmap to match ROI size
            roi_h, roi_w = roi_bgr.shape[:2]
            heatmap_resized = cv2.resize(heatmap_img, (roi_w, roi_h))
            # Place it back in the original image coordinates
            final_img[y1:y2, x1:x2] = heatmap_resized
            
    # Draw bounding box and text
    if detected:
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(final_img, f"YOLO: {confidence:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    color = (0, 255, 0) if verdict == "NORMAL" else (0, 0, 255)
    final_verdict = "PASS" if verdict == "NORMAL" else "FAIL"
    cv2.putText(final_img, f"{final_verdict} (Score: {score:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    final_output_path = PIPELINE_OUT / f"final_{input_path.name}"
    cv2.imwrite(str(final_output_path), final_img)
    log.info(f"Final annotated image saved to {final_output_path}")
    
    # 4. Save JSON
    final_json = {
        "image": str(input_path),
        "yolo": {
            "detected": detected,
            "confidence": confidence,
            "box": [x1, y1, x2, y2]
        },
        "gradcam_path": gradcam_path_str,
        "patchcore": {
            "score": score,
            "threshold": threshold,
            "verdict": verdict
        },
        "final_decision": final_verdict
    }
    
    json_path = PIPELINE_OUT / f"pipeline_{input_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    log.info(f"JSON results saved to {json_path}")
        
    # 5. Print Output Block
    print("\n=====================================")
    print("Industrial AI Quality Inspection")
    print("=====================================\n")
    print("Image:")
    print(input_path.name)
    print("\nYOLO")
    print("----------------------")
    print(f"Detected: {detected}")
    if detected:
        print(f"Confidence: {confidence:.4f}")
    else:
        print("Confidence:")
    print("\nPatchCore")
    print("----------------------")
    print(f"Score: {score:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Verdict: {verdict}")
    print("\nFinal Decision\n")
    print(final_verdict)
    print("\n=====================================")

if __name__ == "__main__":
    main()
