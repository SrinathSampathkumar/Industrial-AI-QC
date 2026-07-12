# 🚀 Industrial AI Quality Control System

> AI-powered Multi-Category Industrial Defect Detection using YOLO and PatchCore

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Status](https://img.shields.io/badge/Status-Ongoing-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📌 Project Overview

Industrial AI Quality Control System is an AI-powered automated inspection platform designed to detect manufacturing defects in industrial products using Computer Vision and Deep Learning.

The system combines:

- YOLO (Object Detection)
- PatchCore (Anomaly Detection)
- FastAPI Backend
- Multi-category Model Registry

to inspect industrial products in real time.

The project currently supports **15 product categories** from the **MVTec AD Dataset** and is being developed as a Final Year B.Tech AI & Data Science project.

---

# 🎯 Objectives

- Detect manufacturing defects automatically
- Reduce manual inspection time
- Improve quality control accuracy
- Support multiple industrial product categories
- Provide scalable AI inspection pipeline

---

# ✨ Features

✔ Multi-category defect inspection

✔ Automatic category routing

✔ PatchCore anomaly detection

✔ Benchmark evaluation

✔ Model Registry

✔ Threshold management

✔ FastAPI backend

✔ Real-time inference support

✔ Modular AI pipeline

✔ Industrial-ready project structure

---

# 🏭 Supported Categories

| Category |
|----------|
| Bottle |
| Cable |
| Capsule |
| Carpet |
| Grid |
| Hazelnut |
| Leather |
| Metal Nut |
| Pill |
| Screw |
| Tile |
| Toothbrush |
| Transistor |
| Wood |
| Zipper |

**Total Categories : 15**

---

# 🧠 AI Pipeline

Input Image

↓

YOLO Detection

↓

Category Router

↓

PatchCore Model Registry

↓

Corresponding PatchCore Model

↓

Anomaly Detection

↓

Threshold Evaluation

↓

PASS / FAIL Decision

↓

FastAPI JSON Response

---

# 📂 Project Structure

```
Industrial-AI-QC/

backend/
    app.py
    app_patchcore.py
    gradcam.py
    live.py

scripts/
    api/
    inference/
    registry/
    training/
    testing/
    utils/

models/

datasets/

reports/

frontend/

README.md

requirements.txt
```

---

# ⚙ Tech Stack

## Programming

- Python 3.12

## AI / ML

- PyTorch
- PatchCore
- Anomalib
- YOLO
- OpenCV
- NumPy
- Pandas
- Scikit-Learn

## Backend

- FastAPI

## Version Control

- Git
- GitHub

---

# 📊 Benchmark Results

| Category | Accuracy | AUROC |
|-----------|----------|--------|
| Bottle | 98.80% | 1.0000 |
| Cable | 82.67% | 0.9751 |
| Capsule | 84.09% | 0.9848 |
| Carpet | 93.16% | 0.9723 |
| Grid | 93.59% | 0.9783 |
| Hazelnut | 100% | 1.0000 |
| Leather | 98.39% | 1.0000 |
| Metal Nut | 99.13% | 1.0000 |
| Pill | 89.82% | 0.9490 |
| Screw | 78.12% | 0.9586 |
| Tile | 99.15% | 1.0000 |
| Toothbrush | 78.57% | 0.9139 |
| Transistor | 81.00% | 0.9892 |
| Wood | 94.94% | 0.9877 |
| Zipper | 94.70% | 0.9850 |

---

# 🏆 Best Performing Categories

- Hazelnut
- Metal Nut
- Tile

---

# ⚠ Categories Under Optimization

- Screw
- Toothbrush
- Transistor

Future improvements include threshold optimization and additional fine-tuning.

---

# 📈 Current Progress

## Completed

- Dataset preparation
- PatchCore training
- Multi-category model support
- Model Registry
- Category Router
- Benchmark pipeline
- Benchmark evaluation
- Threshold optimization
- GitHub integration
- FastAPI backend
- Project structure

---

## In Progress

- YOLO + PatchCore integration
- Defect type classification
- Dashboard development
- Live webcam inference
- Explainable AI (GradCAM)
- IEEE paper writing

---

# 📷 Screenshots

## API (Swagger)

*(Add Swagger screenshot here)*

---

## Live Detection

*(Add webcam inference screenshot here)*

---

## Benchmark Output

*(Add benchmark table screenshot here)*

---

## Project Architecture

*(Add architecture diagram here)*

---

# 🚀 Installation

Clone repository

```bash
git clone https://github.com/SrinathSampathkumar/Industrial-AI-QC.git

cd Industrial-AI-QC
```

Create virtual environment

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Running

Train models

```bash
python scripts/training/train_all_categories.py
```

Benchmark

```bash
python scripts/testing/benchmark_patchcore.py
```

Evaluate

```bash
python scripts/testing/evaluate_benchmark.py
```

Run Backend

```bash
uvicorn backend.app:app --reload
```

---

# 📥 Download Trained Models

The trained PatchCore models are not included in this repository because GitHub has a **100 MB file size limit**.

Download the trained models from:

[**(Add your Google Drive link here)**](https://drive.google.com/file/d/1wc-cvArNTcIOR-zV4urFOmSzgAiCH0rs/view?usp=drive_link)

Extract them into:

```
models/patchcore/
```

---

# 🗺 Roadmap

- [x] PatchCore Training
- [x] Multi-category Support
- [x] Benchmark Evaluation
- [x] Threshold Optimization
- [x] GitHub Repository
- [ ] YOLO Defect Classification
- [ ] REST API Improvements
- [ ] Dashboard
- [ ] Live Camera
- [ ] Explainable AI
- [ ] Docker Deployment
- [ ] ONNX Optimization
- [ ] IEEE Publication

---

# 👨‍💻 Author

**Srinath Sampathkumar**

B.Tech Artificial Intelligence & Data Science

SRM Valliammai Engineering College

Final Year Project (2026–2027)

GitHub:

https://github.com/SrinathSampathkumar

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.