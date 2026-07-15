import os
import shutil
import json
import base64
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse
import sys

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from scripts.inference.yolo_patchcore_pipeline import HybridInspectionPipeline
from scripts.inference.report_generator import ReportGenerator

# Define directories
BACKEND_DIR = BASE_DIR / "backend"
UPLOADS_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Global pipeline state
ai_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading AI models into memory...")
    ai_services["pipeline"] = HybridInspectionPipeline()
    ai_services["report_gen"] = ReportGenerator()
    print("Models loaded successfully.")
    yield
    # Shutdown
    ai_services.clear()

app = FastAPI(title="Industrial AI QC API", lifespan=lifespan)

@app.post("/inspect")
async def inspect_image(
    category: str = Query(..., description="Product Category (e.g., bottle, cable)"),
    file: UploadFile = File(...)
):
    try:
        # 1. Save uploaded image temporarily
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Run inference in memory
        pipeline: HybridInspectionPipeline = ai_services["pipeline"]
        report_gen: ReportGenerator = ai_services["report_gen"]
        
        result_dict = pipeline.inspect(
            image_path=str(file_path),
            category=category,
            output_dir=str(RESULTS_DIR / "heatmaps")
        )
        
        if result_dict.get("status") == "ERROR":
            raise HTTPException(status_code=500, detail=result_dict.get("message"))
            
        # 3. Generate Markdown Report
        report_path = report_gen.generate(result_dict, str(file_path))
        
        # 4. Base64 encode GradCAM if available
        heatmap_base64 = None
        pc_data = result_dict.get("patchcore", {})
        gradcam_path = pc_data.get("heatmap_path")
        
        if gradcam_path and os.path.exists(gradcam_path):
            with open(gradcam_path, "rb") as img_f:
                heatmap_base64 = base64.b64encode(img_f.read()).decode('utf-8')
                
        # 5. Build response matching old API contract
        response_data = {
            "filename": file.filename,
            "category": category,
            "defect_type": result_dict["defect_classification"],
            "confidence": result_dict["confidence_score"],
            "anomaly_score": pc_data.get("normalized_score", 0.0),
            "status": result_dict["status"],
            "report_path": report_path
        }
        
        if heatmap_base64 is not None:
            response_data["heatmap_base64"] = heatmap_base64
            
        # Save JSON result
        result_save_path = RESULTS_DIR / f"result_{file_path.stem}.json"
        with open(result_save_path, "w") as f:
            json.dump(response_data, f, indent=4)
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

