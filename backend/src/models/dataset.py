"""
backend/src/models/dataset.py

PyTorch Dataset and Collation implementations for Speech Emotion Recognition.
"""

import soundfile as sf
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any, List

from src.models.config import EMOTION_LABELS

class SpeechEmotionDataset(Dataset):
    """
    Dataset representing preprocessed audio files mapped to their target labels.
    """
    def __init__(self, manifest_path: Path):
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
            
        self.df = pd.read_csv(manifest_path)
        self.label_map = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
        
    def __len__(self) -> int:
        return len(self.df)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        file_path = row["processed_file_path"]
        emotion = row["emotion"]
        
        # Load audio (already preprocessed to 16kHz mono PCM)
        speech, sr = sf.read(file_path)
        
        return {
            "input_values": speech,
            "label": self.label_map[emotion]
        }

class SpeechDataCollator:
    """
    Data collator that dynamically pads waveforms in a batch to the maximum sequence length.
    """
    def __init__(self, processor: Any):
        self.processor = processor
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_values = [item["input_values"] for item in batch]
        labels = [item["label"] for item in batch]
        
        # Standard huggingface processor pads waveforms and converts to torch tensors
        features = self.processor(
            input_values,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )
        
        # Map labels into torch tensors
        features["labels"] = torch.tensor(labels, dtype=torch.long)
        return features
