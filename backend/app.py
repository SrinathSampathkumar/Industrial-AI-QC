import os
import shutil
import json
import subprocess
import sys
import base64
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Industrial AI QC API")

# Define directories
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
UPLOADS_DIR = BACKEND_DIR / "uploads"
RESULTS_DIR = BACKEND_DIR / "results"

# Ensure directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/inspect")
async def inspect_image(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded image temporarily
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Call the YOLO + PatchCore pipeline
        pipeline_script = BASE_DIR / "scripts" / "yolo_patchcore_pipeline.py"
        python_exe = sys.executable if "venv" not in sys.executable else sys.executable # wait, sys.executable is usually right, but let's just use "python" or sys.executable.
        
        # We need to run the pipeline.
        result = subprocess.run(
            [sys.executable, str(pipeline_script), "--input", str(file_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": "Pipeline execution failed", "details": result.stderr}
            )
            
        # 3. Read pipeline output
        # The pipeline saves the output to BASE_DIR / "results" / "pipeline_output" / f"pipeline_{file_path.stem}.json"
        pipeline_output_dir = BASE_DIR / "results" / "pipeline_output"
        json_path = pipeline_output_dir / f"pipeline_{file_path.stem}.json"
        
        if not json_path.exists():
            return JSONResponse(
                status_code=500,
                content={"error": "Pipeline output JSON not found", "details": result.stdout}
            )
            
        with open(json_path, "r") as f:
            pipeline_data = json.load(f)
            
        # Extract fields
        yolo_data = pipeline_data.get("yolo", {})
        pc_data = pipeline_data.get("patchcore", {})
        
        confidence = yolo_data.get("confidence", 0.0)
        anomaly_score = pc_data.get("score", 0.0)
        status = pipeline_data.get("final_decision", "UNKNOWN")
        
        defect_type = "None" if status == "PASS" else "Anomaly"
        
        # Base64 encode GradCAM if available
        gradcam_path = pipeline_data.get("gradcam_path")
        heatmap_base64 = None
        if gradcam_path and os.path.exists(gradcam_path):
            with open(gradcam_path, "rb") as img_f:
                heatmap_base64 = base64.b64encode(img_f.read()).decode('utf-8')
        
        # Build response
        response_data = {
            "filename": file.filename,
            "defect_type": defect_type,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "status": status
        }
        if heatmap_base64 is not None:
            response_data["heatmap_base64"] = heatmap_base64
        
        # Save inference results inside backend/results/
        result_save_path = RESULTS_DIR / f"result_{file_path.stem}.json"
        with open(result_save_path, "w") as f:
            json.dump(response_data, f, indent=4)
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
