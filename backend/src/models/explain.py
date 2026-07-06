"""
backend/src/models/explain.py

Explainable AI (XAI) engine for Speech Emotion Recognition.
Extracts final self-attention layer weights from Wav2Vec2 and plots them as an overlay on the audio waveform.
"""

import io
import base64
import torch
import numpy as np
import matplotlib
# Use non-interactive backend for server environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from pathlib import Path
from typing import Dict, Any

from src.models.config import MODEL_NAME, SAMPLE_RATE

def generate_attention_map(waveform: np.ndarray, model: Any, processor: Any) -> str:
    """
    Passes waveform through Wav2Vec2, extracts final self-attention weights,
    maps attention to timeline, and generates a base64-encoded plot.
    """
    try:
        device = next(model.parameters()).device
        
        # 1. Run inference & extract attentions
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        # Forward pass with attention tracking
        with torch.no_grad():
            outputs = model(input_values, output_attentions=True)
            
        # Extract attention tuple: 12 elements (one per layer)
        # Final layer index = -1
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        if not attentions:
            raise ValueError("Attentions are empty. Check if model supports output_attentions=True.")
            
        final_layer_att = attentions[-1][0].cpu()  # Get batch index 0, shape: (num_heads, seq_len, seq_len)
        
        # 2. Compute Salience Score
        # Average attention weight across all heads
        mean_att = final_layer_att.mean(dim=0).numpy()  # (seq_len, seq_len)
        # Sum/average attention received by each target frame
        salience = mean_att.mean(axis=0)  # (seq_len,)
        
        # Max-normalize the salience curve for visualization
        if np.max(salience) > 0:
            salience = salience / (np.max(salience) + 1e-8)
            
        # 3. Align Timeline
        # Wav2Vec2 downsamples 16kHz audio by factor of 320 (20ms frames)
        time_axis_wav = np.arange(len(waveform)) / SAMPLE_RATE
        time_axis_att = np.linspace(0, len(waveform) / SAMPLE_RATE, len(salience))
        
        # 4. Generate Plot
        fig, ax1 = plt.subplots(figsize=(10, 3.5), dpi=100)
        
        # Set light minimalist style
        fig.patch.set_facecolor('#fafafa')
        ax1.set_facecolor('#ffffff')
        
        # Plot raw waveform (light gray line)
        ax1.plot(time_axis_wav, waveform, color='#d1d5db', alpha=0.8, label="Audio Signal")
        ax1.set_xlabel("Time (seconds)", fontsize=9, color='#4b5563')
        ax1.set_ylabel("Amplitude", fontsize=9, color='#4b5563')
        ax1.tick_params(axis='both', colors='#4b5563', labelsize=8)
        
        # Instantiate second y-axis for attention weight
        ax2 = ax1.twinx()
        # Smooth attention plot using interpolation
        # Color: Stripe/Vercel purple accent
        ax2.fill_between(time_axis_att, salience, color='#6366f1', alpha=0.25, label="Model Attention")
        ax2.plot(time_axis_att, salience, color='#4f46e5', linewidth=1.5)
        ax2.set_ylabel("Attention Salience", fontsize=9, color='#4f46e5')
        ax2.tick_params(axis='y', colors='#4f46e5', labelsize=8)
        
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        
        plt.title("Acoustic Temporal Attention Highlight Map", fontsize=10, fontweight='bold', color='#1f2937', pad=12)
        fig.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        # Encode to Base64
        base64_str = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"XAI saliency generation failed: {str(e)}")
        return ""
