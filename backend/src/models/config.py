"""
backend/src/models/config.py

Centralized configuration file for training hyperparameters, paths, and target labels.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
OPTIMIZED_DIR = BASE_DIR / "models" / "optimized"

# Ensure directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)

# Dataset & Preprocessing parameters
SAMPLE_RATE = 16000
EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprised"]
NUM_LABELS = len(EMOTION_LABELS)

# Model configuration
MODEL_NAME = "facebook/wav2vec2-base"

# Fine-tuning Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 10
WEIGHT_DECAY = 0.01

# Explainability Target
XAI_LAYER_TARGET = "wav2vec2.encoder.layers.11"
