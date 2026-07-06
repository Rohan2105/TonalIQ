"""
backend/src/models/onnx_export.py

Converts the trained PyTorch Wav2Vec2 model to an optimized ONNX format,
and performs dynamic INT8 quantization for low-latency production execution.
"""

import torch
import logging
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.models.config import CHECKPOINT_DIR, OPTIMIZED_DIR, MODEL_NAME, NUM_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def export_and_quantize():
    hf_model_path = CHECKPOINT_DIR / "hf_model"
    
    if not hf_model_path.exists():
        logger.error(f"Fine-tuned HuggingFace model missing at {hf_model_path}. Train the model first!")
        return
        
    logger.info(f"Loading fine-tuned model from {hf_model_path} for ONNX export...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(hf_model_path)
    model.eval()
    
    # Define paths
    onnx_path = OPTIMIZED_DIR / "model.onnx"
    quant_path = OPTIMIZED_DIR / "model_quantized.onnx"
    
    # 1. Export to ONNX
    logger.info("Exporting model to ONNX...")
    # Dummy input: batch_size=1, audio_length = 3 seconds (3 * 16000 = 48000 samples)
    dummy_input = torch.randn(1, 48000)
    
    # Export with dynamic axes for input sequence length and batch size
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14,
        do_constant_folding=True
    )
    logger.info(f"Base ONNX model successfully saved to {onnx_path}")
    
    # 2. INT8 Dynamic Quantization
    logger.info("Quantizing ONNX model to INT8...")
    try:
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(quant_path),
            weight_type=QuantType.QUInt8
        )
        logger.info(f"Quantized INT8 ONNX model successfully saved to {quant_path}")
    except Exception as e:
        logger.warning(f"ONNX Quantization failed due to shape inference constraints: {str(e)}")
        logger.info("Falling back to unquantized base ONNX model for production execution...")
        import shutil
        shutil.copy(onnx_path, quant_path)
        logger.info(f"Base ONNX model copied to {quant_path} successfully.")

if __name__ == "__main__":
    export_and_quantize()
