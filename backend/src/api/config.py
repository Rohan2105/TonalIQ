"""
backend/src/api/config.py

Configuration settings for the FastAPI application, logging, and inference engines.
"""

import os
from pathlib import Path

# Base Paths
API_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "inference.log"

# Make sure logs directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Model Paths
ONNX_MODEL_PATH = BASE_DIR / "models" / "optimized" / "model_quantized.onnx"
HF_MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "hf_model"
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"

# API Server Settings
HOST = "0.0.0.0"
PORT = 8000
ALLOWED_ORIGINS = ["*"]
