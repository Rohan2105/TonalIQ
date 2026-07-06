# TonalIQ: Production-Grade Decoupled Speech Emotion Recognition (SER) System

TonalIQ is a speech emotion recognition system that classifies vocal tone in emotional categories using audio feature extraction and machine learning. It processes raw audio input to detect emotions like happiness, anger, and sadness, enabling applications in customer support analytics, mental health screening, and human-computer interaction research.

A production-grade, highly optimized Speech Emotion Recognition (SER) system built from scratch using Python, FastAPI, ONNX Runtime, and a minimalist vanilla web frontend. 

The system leverages a fine-tuned self-supervised learning (SSL) Transformer model (**Wav2Vec2**) to classify human vocal speech into seven distinct emotions (*neutral, happy, sad, angry, fear, disgust, surprised*). It features **real-time WebSocket audio streaming**, **temporal self-attention explainability maps**, and **decoupled microservice architecture**.

---

## 🏗️ Project Architecture Layout

The codebase is organized into two independent, cloud-ready folders at the root level:

```text
speech-emotion-recognition/
├── frontend/                      # 🌐 Frontend Microservice (Lightweight UI - Vercel)
│   ├── index.html                 # Minimalist light-theme layout
│   ├── style.css                  # Clean Stone-colored typography & emotion gauges
│   └── app.js                     # Web Audio API 16kHz PCM stream node & REST client
└── backend/                       # 🐍 Machine Learning API (AWS/Render/Docker Container)
    ├── pyproject.toml             # uv package manager dependency configuration
    ├── README.md                  # Backend placeholder readme
    ├── data/                      # Local dataset caches (gitignored)
    ├── models/                    # Training checkpoints and optimized ONNX models (gitignored)
    ├── logs/                      # Latency and telemetry logs (gitignored)
    └── src/                       # Source code package
        ├── pipeline/              # Ingestion & Preprocessing (Noisereduce, Trimming)
        ├── models/                # Fine-tuning, ONNX converter, and XAI Attention mapper
        └── api/                   # Headless FastAPI API layer with strict CORS middleware
```

---

## 🚀 Quick Start Guide

### Prerequisite: Install Python 3.11 & uv
Ensure Python 3.11 is installed. We utilize the ultra-fast Python package installer `uv`. To install `uv` (if not already present):
```powershell
pip install uv
```

### 1. Initialize Backend Environment
Navigate to the `backend/` directory, create a virtual environment, and install dependencies in editable mode:
```powershell
cd backend
uv venv
uv pip install -e .
```

### 2. Programmatic Dataset Acquisition
Run the automated downloader to retrieve RAVDESS and TESS from the Hugging Face Hub cache and write them locally:
```powershell
.venv\Scripts\python.exe -m src.pipeline.download_data
```

### 3. Preprocess & Partition Data
Clean the signals (denoising + silence trimming + normalization) and split them into 70/15/15 stratified, speaker-isolated datasets:
```powershell
.venv\Scripts\python.exe -m src.pipeline.preprocess
```

### 4. Fine-Tune Wav2Vec2
Train the classification head on the preprocessed samples. If running on CPU, the script automatically limits training to a subset of 32 samples for 1 epoch to verify the pipeline compiles instantly:
```powershell
.venv\Scripts\python.exe -m src.models.train
```

### 5. Export and Quantize ONNX
Convert the PyTorch fine-tuned model checkpoint into optimized ONNX format for low-latency inference:
```powershell
.venv\Scripts\python.exe -m src.models.onnx_export
```

### 6. Run the FastAPI Backend Server
Boot up the headless production API server:
```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
```

### 7. Run the Minimalist Frontend Dashboard
Since the frontend is fully decoupled, you can run a local web server (e.g., in port 3000) or simply double-click `frontend/index.html` to open it in your browser:
```powershell
cd ../frontend
python -m http.server 3000
```
Open `http://localhost:3000` in your web browser.

---

## 🔬 Running Automated Verification Tests

Verify the entire system (preprocessing algorithms, FastAPI lifespans, CORS middleware, and ONNX session loads) by running `pytest` in the `backend/` folder:
```powershell
.venv\Scripts\python.exe -m pytest ../tests/
```
All tests will pass successfully.
