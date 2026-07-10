"""
Industrial AI QC API (PatchCore Version)
========================================
FastAPI backend for multi-category PatchCore inspection.

Endpoints
---------
GET  /
GET  /models/status
POST /inspect?category=<category>
"""

from pathlib import Path
import shutil

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from scripts.registry.model_registry import ModelRegistry
from scripts.api.category_router import route_inspection

# -------------------------------------------------------
# FastAPI
# -------------------------------------------------------

app = FastAPI(
    title="Industrial AI QC API",
    version="2.0",
    description="Multi-category Industrial Defect Detection using PatchCore"
)

registry = ModelRegistry()

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Root
# -------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Industrial AI QC API",
        "version": "2.0",
        "status": "running"
    }

# -------------------------------------------------------
# Model Status
# -------------------------------------------------------

@app.get("/models/status")
def model_status():

    status = {}

    for category in registry.list_available():
        status[category] = registry.is_trained(category)

    return JSONResponse(content=status)

# -------------------------------------------------------
# Inspection
# -------------------------------------------------------

@app.post("/inspect")
async def inspect_image(
    category: str = Query(..., description="Product Category"),
    file: UploadFile = File(...)
):

    try:

        image_path = UPLOAD_DIR / file.filename

        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = route_inspection(
            image_path=str(image_path),
            category_key=category
        )

        return JSONResponse(content=result)

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )