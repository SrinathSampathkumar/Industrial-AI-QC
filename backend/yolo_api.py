"""
YOLO Inspection API Router
==========================

Exposes a /yolo/inspect endpoint that supports switching between
the pretrained YOLO model and the fine-tuned MSME defect model.

Endpoint
--------
POST /yolo/inspect?model=pretrained|finetuned&category=<category>

Model Selection
---------------
  pretrained  →  yolo11n.pt  (general-purpose, default)
  finetuned   →  runs/detect/runs/msme_defect_detection-3/weights/best.pt

Weight paths are read from configs/yolo_models.yaml so they are
defined in exactly one place.

Author: Srinath
"""

import shutil
from pathlib import Path

import yaml
from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# Project root & config
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "yolo_models.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _YOLO_MODELS: dict = yaml.safe_load(_f)

_PRETRAINED_PATH = str(_PROJECT_ROOT / _YOLO_MODELS["pretrained"])
_FINETUNED_PATH = str(_PROJECT_ROOT / _YOLO_MODELS["finetuned"])

_UPLOAD_DIR = _PROJECT_ROOT / "backend" / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Import YOLODetector (relative resolution handled internally)
# ------------------------------------------------------------------

import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.inference.yolo_detector import YOLODetector

# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(prefix="/yolo", tags=["YOLO"])


def _resolve_weights(model_choice: str) -> str:
    """Return the absolute weights path for the requested model choice."""
    if model_choice == "finetuned":
        return _FINETUNED_PATH
    return _PRETRAINED_PATH  # default: pretrained


@router.get("/models")
def list_yolo_models():
    """List available YOLO model choices and their resolved paths."""
    return {
        "pretrained": {
            "key": "pretrained",
            "path": _PRETRAINED_PATH,
            "exists": Path(_PRETRAINED_PATH).exists(),
        },
        "finetuned": {
            "key": "finetuned",
            "path": _FINETUNED_PATH,
            "exists": Path(_FINETUNED_PATH).exists(),
        },
    }


@router.post("/inspect")
async def yolo_inspect(
    category: str = Query(..., description="MVTec category (e.g. bottle, cable)"),
    model: str = Query(
        "pretrained",
        description="Which YOLO weights to use: 'pretrained' or 'finetuned'",
    ),
    file: UploadFile = File(...),
):
    """
    Run YOLO defect detection on an uploaded image.

    Parameters
    ----------
    category : str
        The product category key (must match a config in yolo_configs/).
    model : str
        'pretrained' loads yolo11n.pt; 'finetuned' loads best.pt.
    file : UploadFile
        The image to inspect.
    """
    try:
        # Validate model choice
        if model not in ("pretrained", "finetuned"):
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "model must be 'pretrained' or 'finetuned'.",
                },
            )

        # Save uploaded file
        image_path = _UPLOAD_DIR / file.filename
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Resolve weights path
        weights_path = _resolve_weights(model)

        if not Path(weights_path).exists():
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": f"Weights file not found: {weights_path}",
                },
            )

        # Run inference
        detector = YOLODetector(model_path=weights_path)
        result = detector.detect(
            image_path=str(image_path),
            category=category,
        )

        return JSONResponse(
            content={
                "status": "success",
                "model_used": model,
                "weights": weights_path,
                **result,
            }
        )

    except FileNotFoundError as exc:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )
