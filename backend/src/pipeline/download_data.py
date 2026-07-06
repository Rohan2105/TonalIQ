"""
backend/src/pipeline/download_data.py

Automated dataset downloader for RAVDESS and TESS.
Loads audio arrays from Hugging Face Datasets and saves them locally as WAV files under data/raw/.
Bypasses slow/offline institutional servers.
"""

import logging
import soundfile as sf
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Base directories
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"

def download_and_save_ravdess():
    ravdess_dest = RAW_DIR / "ravdess"
    ravdess_dest.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading RAVDESS dataset from HuggingFace Hub (xbgoose/ravdess)...")
    ds = load_dataset('xbgoose/ravdess', split='train', trust_remote_code=True)
    
    logger.info(f"Writing {len(ds)} RAVDESS files to raw directory: {ravdess_dest}")
    for idx, item in enumerate(ds):
        filepath_str = item["audio"]["path"]
        # Extract filename (e.g. 03-01-05-01-02-01-16.wav)
        filename = Path(filepath_str).name
        
        # Audio parameters
        array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        dest_path = ravdess_dest / filename
        sf.write(dest_path, array, sr, subtype='PCM_16')
        
        if (idx + 1) % 200 == 0:
            logger.info(f"Saved {idx + 1}/{len(ds)} RAVDESS files...")

def download_and_save_tess():
    tess_dest = RAW_DIR / "tess"
    tess_dest.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading TESS dataset from HuggingFace Hub (myleslinder/tess)...")
    ds = load_dataset('myleslinder/tess', split='train', trust_remote_code=True)
    
    logger.info(f"Writing {len(ds)} TESS files to raw directory: {tess_dest}")
    for idx, item in enumerate(ds):
        filepath_str = item["audio"]["path"]
        # Extract filename (e.g. OAF_back_angry.wav)
        filename = Path(filepath_str).name
        
        # Audio parameters
        array = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        
        dest_path = tess_dest / filename
        sf.write(dest_path, array, sr, subtype='PCM_16')
        
        if (idx + 1) % 400 == 0:
            logger.info(f"Saved {idx + 1}/{len(ds)} TESS files...")

def main():
    logger.info("Initializing automated acquisition pipeline via Hugging Face...")
    try:
        download_and_save_ravdess()
    except Exception as e:
        logger.error(f"RAVDESS download pipeline failed: {str(e)}")
        raise e
        
    try:
        download_and_save_tess()
    except Exception as e:
        logger.error(f"TESS download pipeline failed: {str(e)}")
        raise e
        
    logger.info("Dataset acquisition completed successfully.")

if __name__ == "__main__":
    main()
