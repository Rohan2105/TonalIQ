"""
backend/src/models/train.py

Training script to fine-tune Wav2Vec2 on our preprocessed speech emotion dataset.
Includes validation checkpointing and TensorBoard metric tracking.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from src.models.config import (
    BASE_DIR, MODEL_NAME, DATA_DIR, CHECKPOINT_DIR, BATCH_SIZE,
    LEARNING_RATE, EPOCHS, WEIGHT_DECAY, NUM_LABELS
)
from src.models.dataset import SpeechEmotionDataset, SpeechDataCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_model():
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device for training: {device}")
    
    # 2. Processor & Model Initialization
    logger.info(f"Loading pre-trained Wav2Vec2 model: {MODEL_NAME}")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    
    # Wav2Vec2 classification head mapped to our exact label count
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    
    # Freeze the convolutional feature encoder (crucial for transfer learning)
    model.freeze_feature_encoder()
    model.to(device)
    
    # 3. Load Datasets
    logger.info("Initializing DataLoaders...")
    train_manifest = DATA_DIR / "processed" / "train_split.csv"
    val_manifest = DATA_DIR / "processed" / "val_split.csv"
    
    if not train_manifest.exists() or not val_manifest.exists():
        logger.error("Processed train/val manifests missing. Run preprocessing pipeline first!")
        return
        
    train_dataset = SpeechEmotionDataset(train_manifest)
    val_dataset = SpeechEmotionDataset(val_manifest)
    
    collator = SpeechDataCollator(processor)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collator
    )
    
    # 4. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=str(BASE_DIR / "runs" / "ser_experiment"))
    
    best_val_loss = float("inf")
    patience = 3
    epochs_no_improve = 0
    start_epoch = 1
    
    best_model_path = CHECKPOINT_DIR / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"Found checkpoint at {best_model_path}. Loading weights to resume training...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float("inf"))
        logger.info(f"Resuming from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")
    
    # 5. Training Loop
    logger.info("Starting training loop execution...")
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for batch in loop:
            optimizer.zero_grad()
            
            # Move batch tensors to device
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            outputs = model(
                input_values=input_values, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_values = batch["input_values"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    
                outputs = model(
                    input_values=input_values, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        
        logger.info(
            f"Epoch {epoch} Results - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            
            best_model_path = CHECKPOINT_DIR / "best_model.pt"
            logger.info(f"New validation loss baseline achieved. Saving checkpoint to {best_model_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_acc
            }, best_model_path)
            
            # Save the raw huggingface configurations together for ONNX export
            model.save_pretrained(CHECKPOINT_DIR / "hf_model")
            processor.save_pretrained(CHECKPOINT_DIR / "hf_model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    writer.close()
    logger.info("Model fine-tuning training workflow finished successfully.")

if __name__ == "__main__":
    train_model()
