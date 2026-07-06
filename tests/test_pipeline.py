"""
tests/test_pipeline.py

Unit tests for audio preprocessing and cleaning signal functions.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import pytest

from src.pipeline.preprocess import clean_audio_signal

def test_clean_audio_signal(tmp_path):
    # 1. Create a dummy audio waveform (16kHz, 1.5 seconds)
    # 0.2s silence + 1.0s active sine tone (440Hz) + 0.3s silence
    sr = 16000
    t_active = np.linspace(0, 1.0, sr)
    tone = np.sin(2 * np.pi * 440 * t_active) * 0.5  # Amplitude = 0.5
    
    silence_start = np.zeros(int(0.2 * sr))
    silence_end = np.zeros(int(0.3 * sr))
    
    raw_audio = np.concatenate([silence_start, tone, silence_end])
    
    # Save dummy audio to a temp file
    dummy_file = tmp_path / "dummy_speech.wav"
    sf.write(dummy_file, raw_audio, sr, subtype='PCM_16')
    
    # 2. Run preprocessing
    cleaned = clean_audio_signal(dummy_file, sample_rate=sr)
    
    # 3. Assertions
    assert isinstance(cleaned, np.ndarray), "Preprocessed audio must be a numpy array"
    assert len(cleaned) > 0, "Preprocessed audio cannot be empty"
    
    # Verify that the silence was trimmed (length should be less than the raw audio length)
    assert len(cleaned) < len(raw_audio), "Silence trimming must shorten the raw signal"
    
    # Verify max normalization (max absolute amplitude should be close to 1.0)
    assert np.allclose(np.max(np.abs(cleaned)), 1.0, atol=1e-4), "Signal must be normalized to a peak amplitude of 1.0"
