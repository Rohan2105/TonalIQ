"""
backend/src/pipeline/preprocess.py

Handles raw audio loading, cleaning (noisereduce + silence removal + normalization),
and partitioning into a speaker-isolated, stratified 70/15/15 train/val/test split.
"""

import logging
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import noisereduce as nr
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

def clean_audio_signal(file_path: Path, sample_rate: int = 16000) -> np.ndarray:
    """
    Loads raw audio, applies spectral gating noise reduction, strips leading/trailing
    silence, and max-normalizes amplitude.
    """
    try:
        # Load audio (Wav2Vec2 natively expects 16kHz mono)
        y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        
        # Apply noisereduce spectral gating
        # Using stationary noise reduction (first 500ms or general profile)
        if len(y) > 8000:
            y_denoised = nr.reduce_noise(y=y, sr=sample_rate, prop_decrease=0.85)
        else:
            y_denoised = y
            
        # Trim silence (top_db=20 is standard for room tone)
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)
        
        # Max-normalize the signal amplitude to avoid volume discrepancies
        if len(y_trimmed) > 0:
            return librosa.util.normalize(y_trimmed)
        return librosa.util.normalize(y_denoised)
        
    except Exception as e:
        logger.error(f"Failed preprocessing file {file_path}: {str(e)}")
        raise e

def create_raw_metadata_csv(raw_dir: Path) -> Path:
    """
    Scans the raw directory for RAVDESS and TESS files and compiles a unified raw metadata CSV.
    """
    records = []
    
    # 1. Scan RAVDESS
    ravdess_dir = raw_dir / "ravdess"
    if ravdess_dir.exists():
        logger.info("Scanning RAVDESS directory...")
        # Schema: Actor_XX/Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
        for file_path in ravdess_dir.rglob("*.wav"):
            name_parts = file_path.stem.split("-")
            if len(name_parts) == 7:
                actor_id = int(name_parts[6])
                emotion_code = name_parts[2]
                emotion_map = {
                    "01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
                    "05": "angry", "06": "fear", "07": "disgust", "08": "surprised"
                }
                emotion = emotion_map.get(emotion_code, "unknown")
                records.append({
                    "file_path": str(file_path.resolve()),
                    "source": "ravdess",
                    "speaker": f"ravdess_actor_{actor_id:02d}",
                    "gender": "male" if actor_id % 2 != 0 else "female",
                    "emotion": emotion
                })
                
    # 2. Scan TESS
    tess_dir = raw_dir / "tess"
    if tess_dir.exists():
        logger.info("Scanning TESS directory...")
        # Schema: OAF_word_emotion.wav or YAF_word_emotion.wav
        for file_path in tess_dir.rglob("*.wav"):
            name_parts = file_path.stem.split("_")
            if len(name_parts) >= 3:
                speaker_prefix = name_parts[0].upper()  # OAF or YAF
                emotion_label = name_parts[2].lower()
                emotion_map = {
                    "neutral": "neutral", "happy": "happy", "sad": "sad",
                    "angry": "angry", "fear": "fear", "disgust": "disgust",
                    "ps": "surprised"
                }
                emotion = emotion_map.get(emotion_label, "unknown")
                records.append({
                    "file_path": str(file_path.resolve()),
                    "source": "tess",
                    "speaker": f"tess_{speaker_prefix.lower()}",
                    "gender": "female",
                    "emotion": emotion
                })
                
    df = pd.DataFrame(records)
    # Filter out any unknown emotions
    df = df[df["emotion"] != "unknown"]
    
    meta_path = raw_dir / "combined_metadata.csv"
    df.to_csv(meta_path, index=False)
    logger.info(f"Compiled raw metadata with {len(df)} entries saved at {meta_path}")
    return meta_path

def prepare_speaker_isolated_splits(metadata_csv_path: Path, output_dir: Path) -> None:
    """
    Groups and partitions datasets: speaker-isolated for RAVDESS, stratified split for TESS.
    Saves splits as processed metadata registers and generates preprocessed clean wav files.
    """
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Source metadata file missing at: {metadata_csv_path}")
        
    df = pd.read_csv(metadata_csv_path)
    logger.info(f"Loaded metadata containing {len(df)} records. Partitioning...")
    
    # 1. Partition RAVDESS (Speaker Isolation)
    df_rav = df[df["source"] == "ravdess"].copy()
    if not df_rav.empty:
        # Extract actor ID from speaker name (e.g. ravdess_actor_04 -> 4)
        df_rav["actor_id"] = df_rav["speaker"].apply(lambda s: int(s.split("_")[-1]))
        train_rav = df_rav[df_rav["actor_id"] <= 16]
        val_rav = df_rav[(df_rav["actor_id"] >= 17) & (df_rav["actor_id"] <= 20)]
        test_rav = df_rav[df_rav["actor_id"] >= 21]
    else:
        train_rav, val_rav, test_rav = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # 2. Partition TESS (Stratified 70/15/15)
    df_tess = df[df["source"] == "tess"].copy()
    if not df_tess.empty:
        labels = df_tess["emotion"].values
        # Split off test (15%)
        train_val_tess, test_tess = train_test_split(
            df_tess, test_size=0.15, stratify=labels, random_state=42
        )
        # Split off train/val (15% of total out of remaining 85% is ~17.65%)
        val_ratio = 0.15 / 0.85
        train_tess, val_tess = train_test_split(
            train_val_tess, test_size=val_ratio, stratify=train_val_tess["emotion"].values, random_state=42
        )
    else:
        train_tess, val_tess, test_tess = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # 3. Combine splits
    train_df = pd.concat([train_rav, train_tess], ignore_index=True)
    val_df = pd.concat([rav_val for rav_val in [val_rav, val_tess] if not rav_val.empty], ignore_index=True)
    test_df = pd.concat([test_rav, test_tess], ignore_index=True)
    
    total_len = len(df)
    logger.info("--- Data Split Verification ---")
    logger.info(f"Train Set      : {len(train_df)} rows ({len(train_df)/total_len:.1%})")
    logger.info(f"Validation Set : {len(val_df)} rows ({len(val_df)/total_len:.1%})")
    logger.info(f"Test Set       : {len(test_df)} rows ({len(test_df)/total_len:.1%})")
    
    # 4. Generate Preprocessed Waveforms and manifests
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits_meta = [("train", train_df), ("val", val_df), ("test", test_df)]
    for split_name, split_data in splits_meta:
        processed_records = []
        logger.info(f"Preprocessing & exporting {split_name} split files...")
        
        split_subdir = output_dir / split_name
        split_subdir.mkdir(parents=True, exist_ok=True)
        
        for idx, row in split_data.iterrows():
            src_file = Path(row["file_path"])
            dest_emotion_dir = split_subdir / row["emotion"]
            dest_emotion_dir.mkdir(exist_ok=True)
            
            dest_file = dest_emotion_dir / src_file.name
            try:
                # Clean audio signal
                y_clean = clean_audio_signal(src_file)
                # Save as 16kHz WAV format (PCM 16-bit)
                sf.write(dest_file, y_clean, 16000, subtype='PCM_16')
                
                # Append record with processed path
                processed_records.append({
                    "raw_file_path": row["file_path"],
                    "processed_file_path": str(dest_file.resolve()),
                    "source": row["source"],
                    "speaker": row["speaker"],
                    "gender": row["gender"],
                    "emotion": row["emotion"]
                })
            except Exception as e:
                logger.error(f"Skipping file {src_file} due to processing error: {str(e)}")
                
        # Write split metadata register sheet
        pd.DataFrame(processed_records).to_csv(output_dir / f"{split_name}_split.csv", index=False)
        
    logger.info(f"Preprocessing completed. Processed files & manifests saved to {output_dir}")

def main():
    logger.info("Initializing preprocessing pipeline execution...")
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    
    meta_path = RAW_DATA_DIR / "combined_metadata.csv"
    if not meta_path.exists():
        meta_path = create_raw_metadata_csv(RAW_DATA_DIR)
        
    prepare_speaker_isolated_splits(meta_path, PROCESSED_DATA_DIR)

if __name__ == "__main__":
    main()
