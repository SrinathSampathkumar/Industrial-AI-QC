"""
Live Webcam Inspection Mode
============================
Runs real-time industrial AI quality inspection using the webcam.

Usage:
    python backend/live.py
    python backend/live.py --camera 0
    python backend/live.py --yolo-model yolo11n.pt

Controls:
    Q  — Quit
    S  — Save current frame snapshot to results/gradcam/

Pipeline per frame:
    Webcam frame
        → YOLO detection (bounding box + class + confidence)
        → PatchCore anomaly score
        → GradCAM heatmap overlay (when detection exists)
        → Overlay text: Class / Confidence / Score / PASS/FAIL
"""

import sys
import time
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backend.gradcam import YOLOGradCAM, overlay_heatmap
from inference_patchcore import run_inference_direct

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("live_inspection")

# ── Constants ─────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "patchcore" / "leather"
GRADCAM_OUT = PROJECT_ROOT / "results" / "gradcam"
PIPELINE_TMP = PROJECT_ROOT / "results" / "pipeline_output"


# ── PatchCore loader (loaded once at startup) ─────────────────
def load_patchcore(results_dir: Path, device: torch.device):
    """Load the PatchCore model.pt once and return the torch model + threshold."""
    from torchvision import transforms
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    import json

    model_pt = results_dir / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(f"PatchCore model not found at {model_pt}")

    checkpoint = torch.load(str(model_pt), map_location=device)
    log.info("PatchCore backbone: %s", checkpoint["backbone"])

    torch_model = PatchcoreModel(
        backbone=checkpoint["backbone"],
        layers=checkpoint["layers"],
        num_neighbors=checkpoint["num_neighbors"],
    )
    torch_model.memory_bank = checkpoint["memory_bank"].to(device)
    torch_model.eval()
    torch_model.to(device)

    # Load threshold
    threshold = 0.5
    threshold_path = results_dir / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            td = json.load(f)
        if isinstance(td.get("image_threshold"), float):
            threshold = td["image_threshold"]
    log.info("PatchCore threshold: %.4f", threshold)

    # Image transform matching PatchCore defaults
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return torch_model, threshold, transform


def run_patchcore_on_frame(frame_bgr, torch_model, threshold, transform, device):
    """Run PatchCore on a single BGR frame. Returns (score, verdict, anomaly_map_bgr)."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = torch_model(tensor)

    score = float(output.pred_score[0])
    verdict = "ANOMALOUS" if score > threshold else "NORMAL"

    anomaly_map_bgr = None
    if output.anomaly_map is not None:
        amap = output.anomaly_map[0, 0].cpu().numpy()
        amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heatmap = cv2.applyColorMap((amap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img_resized = cv2.resize(frame_bgr, (256, 256))
        anomaly_map_bgr = cv2.addWeighted(img_resized, 0.5, heatmap, 0.5, 0)

    return score, verdict, anomaly_map_bgr


def draw_overlay(frame, detected, class_name, confidence, score, verdict, has_gradcam=False):
    """Draw HUD overlay text on the frame in place. Returns the annotated frame."""
    h, w = frame.shape[:2]

    is_fail = verdict == "ANOMALOUS"
    status_text = "FAIL" if is_fail else "PASS"
    status_color = (0, 0, 255) if is_fail else (0, 220, 0)   # red / green (BGR)
    accent = (0, 180, 255)   # orange-ish for labels

    # Semi-transparent top banner
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 70), (20, 20, 20), -1)
    frame = cv2.addWeighted(banner, 0.65, frame, 0.35, 0)

    # Status pill (top-right)
    pill_x = w - 140
    cv2.rectangle(frame, (pill_x, 8), (w - 10, 58), status_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (pill_x + 18, 45),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    # Top-left info
    if detected:
        det_text = f"Class: {class_name}  Conf: {confidence:.2f}"
    else:
        det_text = "No detection — full image used"
    cv2.putText(frame, det_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, accent, 2, cv2.LINE_AA)

    score_text = f"Anomaly Score: {score:.3f}"
    cv2.putText(frame, score_text, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    # GradCAM indicator
    if has_gradcam:
        cv2.putText(frame, "GradCAM ON", (w - 140, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1, cv2.LINE_AA)

    # Bottom footer
    footer_y = h - 10
    cv2.putText(frame, "Q=Quit  S=Snapshot", (10, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Industrial AI QC — Live Webcam Inspection")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--yolo-model", default=str(PROJECT_ROOT / "yolo11n.pt"),
                        help="Path to YOLO model weights")
    parser.add_argument("--skip-frames", type=int, default=3,
                        help="Run PatchCore every N frames to maintain FPS (default: 3)")
    parser.add_argument("--gradcam", action="store_true", default=True,
                        help="Show GradCAM overlay (default: enabled)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("Industrial AI QC — Live Inspection Mode")
    log.info("=" * 60)
    log.info("Device     : %s", device)
    log.info("Camera     : %d", args.camera)
    log.info("YOLO model : %s", args.yolo_model)

    # ── Load models once ──────────────────────────────────────
    log.info("Loading YOLO model...")
    yolo_model = YOLO(args.yolo_model)

    log.info("Loading PatchCore model...")
    try:
        pc_model, pc_threshold, pc_transform = load_patchcore(RESULTS_DIR, device)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("Initialising GradCAM...")
    gradcam_engine = YOLOGradCAM(yolo_model)

    GRADCAM_OUT.mkdir(parents=True, exist_ok=True)
    PIPELINE_TMP.mkdir(parents=True, exist_ok=True)

    # ── Open webcam ───────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera %d", args.camera)
        sys.exit(1)

    # Try to set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    log.info("Webcam opened. Press Q to quit, S to save snapshot.")

    # ── State variables ───────────────────────────────────────
    frame_count = 0
    last_score = 0.0
    last_verdict = "NORMAL"
    last_class = "object"
    last_conf = 0.0
    last_detected = False
    display_frame = None
    snapshot_count = 0
    fps_timer = time.time()
    fps_value = 0.0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Failed to grab frame — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        show_gradcam_overlay = False

        # ── FPS calc ──────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_value = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        # ── YOLO (every frame — fast) ─────────────────────────
        yolo_results = yolo_model(frame, verbose=False)
        detected = False
        confidence = 0.0
        class_name = "object"
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]

        if len(yolo_results) > 0 and len(yolo_results[0].boxes) > 0:
            boxes = yolo_results[0].boxes
            best_idx = torch.argmax(boxes.conf).item()
            best_box = boxes[best_idx]
            confidence = float(best_box.conf.item())
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            detected = True
            # Get class name
            cls_id = int(best_box.cls.item())
            names = yolo_model.names
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        last_detected = detected
        last_conf = confidence
        last_class = class_name

        # ── PatchCore (every N frames — heavy) ────────────────
        if frame_count % args.skip_frames == 0 or display_frame is None:
            roi = frame[y1:y2, x1:x2] if detected else frame.copy()
            if roi.size > 0:
                last_score, last_verdict, _ = run_patchcore_on_frame(
                    roi, pc_model, pc_threshold, pc_transform, device
                )

        # ── GradCAM (every N frames when detected) ────────────
        work_frame = frame.copy()
        if detected and args.gradcam and frame_count % args.skip_frames == 0:
            try:
                cam = gradcam_engine.generate(frame)
                if cam is not None:
                    work_frame = overlay_heatmap(frame, cam)
                    show_gradcam_overlay = True
            except Exception as e:
                log.debug("GradCAM error: %s", e)

        # ── Draw bounding box ─────────────────────────────────
        if detected:
            box_color = (0, 0, 220) if last_verdict == "ANOMALOUS" else (0, 220, 0)
            cv2.rectangle(work_frame, (x1, y1), (x2, y2), box_color, 2)

        # ── Draw HUD overlay ──────────────────────────────────
        display_frame = draw_overlay(
            work_frame, last_detected, last_class, last_conf,
            last_score, last_verdict, has_gradcam=show_gradcam_overlay
        )

        # ── FPS counter ───────────────────────────────────────
        cv2.putText(display_frame, f"FPS: {fps_value:.1f}",
                    (display_frame.shape[1] - 130, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)

        # ── Show ──────────────────────────────────────────────
        cv2.imshow("Industrial AI QC — Live Inspection (Q=Quit)", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            log.info("Quit signal received.")
            break
        elif key == ord('s') or key == ord('S'):
            snapshot_count += 1
            snap_path = GRADCAM_OUT / f"snapshot_{snapshot_count:04d}.png"
            cv2.imwrite(str(snap_path), display_frame)
            log.info("Snapshot saved: %s", snap_path)

    cap.release()
    cv2.destroyAllWindows()
    log.info("Live inspection ended.")


if __name__ == "__main__":
    main()
