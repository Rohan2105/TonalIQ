"""
backend/src/api/main.py

FastAPI production server implementing asynchronous WebSocket and REST API endpoints
for low-latency speech emotion recognition.
"""

import os
import io
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import soundfile as sf
import onnxruntime as ort
import torch
torch.set_num_threads(1)
import torch.nn.functional as F
from pydub import AudioSegment
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from src.api.config import (
    ONNX_MODEL_PATH, HF_MODEL_PATH, DEFAULT_MODEL_NAME, LOG_FILE, ALLOWED_ORIGINS
)
from src.models.config import EMOTION_LABELS, SAMPLE_RATE
from src.models.explain import generate_attention_map
from src.pipeline.preprocess import clean_audio_signal

# Configure logging to console and to file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add file handler specifically for inference tracking
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

app = FastAPI(title="Decoupled Speech Emotion Recognition Engine")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & processor states
ort_session = None
pytorch_model = None
processor = None

@app.on_event("startup")
async def startup_event():
    """Loads processor and model on startup with clean fallbacks."""
    global ort_session, pytorch_model, processor
    
    logger.info("Initializing inference models and processor...")
    
    # 1. Load HF Processor
    try:
        if HF_MODEL_PATH.exists():
            logger.info(f"Loading processor from checkpoint: {HF_MODEL_PATH}")
            processor = Wav2Vec2Processor.from_pretrained(HF_MODEL_PATH)
        else:
            logger.info(f"Loading default processor: {DEFAULT_MODEL_NAME}")
            processor = Wav2Vec2Processor.from_pretrained(DEFAULT_MODEL_NAME)
    except Exception as e:
        logger.critical(f"Failed loading processor: {str(e)}")
        raise e

    # 2. Load PyTorch Model (used for explainability and fallback)
    try:
        if HF_MODEL_PATH.exists():
            logger.info(f"Loading PyTorch model from checkpoint: {HF_MODEL_PATH}")
            pytorch_model = Wav2Vec2ForSequenceClassification.from_pretrained(HF_MODEL_PATH, low_cpu_mem_usage=True)
        else:
            logger.info(f"Loading default PyTorch model (zero-shot fallback): {DEFAULT_MODEL_NAME}")
            pytorch_model = Wav2Vec2ForSequenceClassification.from_pretrained(DEFAULT_MODEL_NAME, low_cpu_mem_usage=True)
        pytorch_model.eval()
    except Exception as e:
        logger.error(f"Failed loading PyTorch model: {str(e)}")

    # 3. Load ONNX Session (production high-performance inference)
    if ONNX_MODEL_PATH.exists():
        try:
            logger.info(f"Initializing optimized ONNX session: {ONNX_MODEL_PATH}")
            ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH))
        except Exception as e:
            logger.error(f"Failed initializing ONNX session: {str(e)}")
            ort_session = None
    else:
        logger.warning(f"ONNX model missing at {ONNX_MODEL_PATH}. Running on PyTorch fallback.")

def run_model_inference(waveform: np.ndarray) -> np.ndarray:
    """Runs input values through ONNX or PyTorch and returns class probabilities."""
    global ort_session, pytorch_model, processor
    
    if ort_session is not None:
        # ONNX Inference path
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="np")
        input_values = inputs.input_values.astype(np.float32)
        
        onnx_inputs = {"input_values": input_values}
        logits = ort_session.run(None, onnx_inputs)[0]
        
        # Softmax over logits (numpy equivalent)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs[0]
    elif pytorch_model is not None:
        # PyTorch Fallback path
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            outputs = pytorch_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        return probs[0].numpy()
    else:
        raise RuntimeError("No active model engine loaded.")

@app.get("/health")
async def health_check():
    """Returns the system status and which inference engine is active."""
    return {
        "status": "online",
        "onnx_loaded": ort_session is not None
    }

@app.post("/api/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """
    Ingests an uploaded audio file, transcodes it, runs model predictions,
    and returns emotion probability distributions alongside attention overlays.
    """
    start_time = time.time()
    logger.info(f"File classification request received: {file.filename}")
    
    try:
        # 1. Read uploaded file bytes
        file_bytes = await file.read()
        
        # 2. Transcode to 16kHz PCM WAV in memory
        try:
            audio = AudioSegment.from_file(io.BytesIO(file_bytes))
            audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
            
            # Extract samples as float32
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            if audio.sample_width == 2:
                samples = samples / 32768.0
            elif audio.sample_width == 4:
                samples = samples / 2147483648.0
        except Exception as e:
            logger.error(f"Transcoding failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid audio format or transcoding failure: {str(e)}")
            
        # 3. Clean and Denoise Audio Signal
        # Save temp file for librosa.load / clean_audio_signal wrapper
        temp_wav = Path(f"temp_{int(time.time())}.wav")
        sf.write(temp_wav, samples, SAMPLE_RATE, subtype='PCM_16')
        
        try:
            clean_samples = clean_audio_signal(temp_wav, sample_rate=SAMPLE_RATE)
        finally:
            if temp_wav.exists():
                os.remove(temp_wav)
                
        # 4. Execute Model Inference
        probs = run_model_inference(clean_samples)
        dominant_idx = int(np.argmax(probs))
        dominant_emotion = EMOTION_LABELS[dominant_idx]
        
        # Format mapping dict
        emotion_probs = {emotion: float(probs[idx]) for idx, emotion in enumerate(EMOTION_LABELS)}
        
        # 5. Generate Attention Explainability Graph (requires PyTorch)
        explain_b64 = ""
        if pytorch_model is not None:
            explain_b64 = generate_attention_map(clean_samples, pytorch_model, processor)
            
        latency = (time.time() - start_time) * 1000
        
        # Logs telemetry metrics to logs/inference.log
        logger.info(
            f"FILE_INFERENCE | File: {file.filename} | "
            f"Latency: {latency:.2f}ms | "
            f"Dominant: {dominant_emotion} | "
            f"Confidence: {probs[dominant_idx]:.4f}"
        )
        
        return {
            "dominant_emotion": dominant_emotion,
            "probabilities": emotion_probs,
            "explainability_graph": explain_b64
        }
        
    except Exception as e:
        logger.error(f"File prediction request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference exception: {str(e)}")

@app.websocket("/api/predict/stream")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Asynchronous WebSocket endpoint processing real-time raw audio binary float32 chunks.
    Maintains a sliding window buffer and sends classifications back in real-time.
    """
    await websocket.accept()
    logger.info("Real-time audio streaming connection established.")
    
    # 3-second buffer at 16kHz is 48,000 samples
    window_size = 3 * SAMPLE_RATE
    # Step size of 0.5s is 8,000 samples
    step_size = int(0.5 * SAMPLE_RATE)
    
    audio_buffer = []
    
    try:
        while True:
            # Receive binary chunk from browser AudioContext node (float32 array bytes)
            data = await websocket.receive_bytes()
            
            # Convert raw bytes back to float32
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer.extend(chunk)
            
            # Run inference when sliding window fills
            if len(audio_buffer) >= window_size:
                # Take current window
                window_data = np.array(audio_buffer[:window_size], dtype=np.float32)
                
                # Slide buffer window by shifting step size
                audio_buffer = audio_buffer[step_size:]
                
                # Check silence/energy: skip prediction if signal is too quiet
                rms = np.sqrt(np.mean(window_data**2))
                if rms < 0.01:
                    mock_response = {
                        "dominant_emotion": "Silence",
                        "probabilities": {emotion: 0.0 for emotion in EMOTION_LABELS}
                    }
                    await websocket.send_text(json.dumps(mock_response))
                    continue
                
                start_inference = time.time()
                # Run prediction
                probs = run_model_inference(window_data)
                latency = (time.time() - start_inference) * 1000
                
                dominant_idx = int(np.argmax(probs))
                dominant_emotion = EMOTION_LABELS[dominant_idx]
                emotion_probs = {emotion: float(probs[idx]) for idx, emotion in enumerate(EMOTION_LABELS)}
                
                # Log streaming performance metrics
                logger.info(
                    f"STREAM_INFERENCE | Latency: {latency:.2f}ms | "
                    f"Dominant: {dominant_emotion} | "
                    f"Confidence: {probs[dominant_idx]:.4f}"
                )
                
                # Send text response payload back
                response = {
                    "dominant_emotion": dominant_emotion,
                    "probabilities": emotion_probs
                }
                await websocket.send_text(json.dumps(response))
                
    except WebSocketDisconnect:
        logger.info("Real-time streaming connection closed cleanly by client.")
    except Exception as e:
        logger.error(f"WebSocket execution exception interruption: {str(e)}")
        await websocket.close(code=1011)

@app.get("/health")
async def health_check():
    """Simple API status check."""
    return {
        "status": "healthy",
        "onnx_loaded": ort_session is not None,
        "pytorch_loaded": pytorch_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    from src.api.config import HOST, PORT
    logger.info(f"Starting server host: {HOST} port: {PORT}")
    uvicorn.run("src.api.main:app", host=HOST, port=PORT, reload=True)
