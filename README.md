Industrial AI Quality Control System
AI-Based Automated Industrial Defect Detection using YOLO + PatchCore

An end-to-end industrial quality inspection system that combines object detection (YOLO) and anomaly detection (PatchCore) for real-time defect detection across multiple industrial product categories from the MVTec AD dataset.

Project Overview

This project aims to build an intelligent quality control system capable of:

Detecting industrial products using YOLO.
Automatically selecting the correct PatchCore model.
Detecting unseen manufacturing defects.
Producing anomaly scores and classification results.
Supporting multiple industrial categories.

The project is developed as part of a Final Year AI & Data Science major project.

Features
Multi-category industrial inspection
Automatic category routing
PatchCore anomaly detection
Real-time inference
Benchmark evaluation
Threshold management
Model registry
Heatmap generation
Benchmark reporting
Python 3.12 compatible
Anomalib 2.5 compatible
Supported Categories
Category
Bottle
Cable
Capsule
Carpet
Grid
Hazelnut
Leather
Metal Nut
Pill
Screw
Tile
Toothbrush
Transistor
Wood
Zipper

Total Supported Categories: 15

Project Structure
Industrial-AI-QC/

backend/
    app.py
    app_patchcore.py
    live.py
    gradcam.py

scripts/
    api/
    inference/
    training/
    testing/
    registry/
    utils/
    legacy/

datasets/

models/

reports/

configs/

frontend/

outputs/
Technologies Used
Python 3.12
PyTorch
Anomalib 2.5
PatchCore
YOLO
OpenCV
NumPy
Pandas
Scikit-learn
Matplotlib
FastAPI / Flask
Benchmark Results

Evaluated on all 15 MVTec AD categories.

Category	Accuracy	AUROC
Bottle	98.80%	1.0000
Cable	82.67%	0.9751
Capsule	84.09%	0.9848
Carpet	93.16%	0.9723
Grid	93.59%	0.9783
Hazelnut	100.00%	1.0000
Leather	98.39%	1.0000
Metal Nut	99.13%	1.0000
Pill	89.82%	0.9490
Screw	78.12%	0.9586
Tile	99.15%	1.0000
Toothbrush	78.57%	0.9139
Transistor	81.00%	0.9892
Wood	94.94%	0.9877
Zipper	94.70%	0.9850
Best Performing Categories
Hazelnut
Metal Nut
Tile
Challenging Categories
Screw
Toothbrush
Transistor

These categories will be further improved through additional training and threshold optimization.

Current Project Status
Completed
Dataset preparation
Training pipeline
Batch training (15 categories)
PatchCore model registry
Threshold registry
Benchmark pipeline
Benchmark evaluation
Automatic category loading
Multi-category inference
Git repository setup
GitHub repository setup
In Progress
YOLO + PatchCore integration
REST API improvements
Live webcam inference
Frontend dashboard
Explainable AI (Grad-CAM)
IEEE Paper Writing
Repository
Industrial-AI-QC

GitHub Repository:

https://github.com/SrinathSampathkumar/Industrial-AI-QC
Installation
git clone https://github.com/SrinathSampathkumar/Industrial-AI-QC.git

cd Industrial-AI-QC

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
Run Training
python scripts/training/train_all_categories.py
Benchmark
python scripts/testing/benchmark_patchcore.py
Evaluate Benchmark
python scripts/testing/evaluate_benchmark.py
Model Registry
from scripts.registry.model_registry import ModelRegistry

registry = ModelRegistry()

threshold = registry.get_threshold("bottle")
Future Work
Real-time industrial inspection
Docker deployment
Edge AI optimization
ONNX export
TensorRT optimization
PLC integration
Explainable AI
Industrial dashboard
Production deployment
Author

Srinath Sampathkumar

B.Tech – Artificial Intelligence & Data Science

SRM Valliammai Engineering College

Final Year Major Project (2026–2027)

## Download Trained PatchCore Models

The trained PatchCore models are not included in this repository because GitHub has a 100 MB file limit.

Download the trained models here:

https://drive.google.com/file/d/1wc-cvArNTcIOR-zV4urFOmSzgAiCH0rs/view?usp=drive_link


After downloading, extract the ZIP into:

models/patchcore/