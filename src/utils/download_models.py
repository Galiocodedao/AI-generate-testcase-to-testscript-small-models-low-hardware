"""
Utility for downloading pre-trained models
"""

import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src import config

logger = logging.getLogger(__name__)

def download_models():
    """
    Download pre-trained models defined in the configuration
    """
    logger.info("Downloading pre-trained models")
    
    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Download each model defined in the config
    for model_name, model_config in config.MODEL_CONFIG.items():
        logger.info(f"Downloading model: {model_name} ({model_config['model_name']})")
        
        try:
            # This will download the model if not already cached
            model = SentenceTransformer(model_config['model_name'])
            logger.info(f"Successfully downloaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {str(e)}")

if __name__ == "__main__":
    from src.utils.logger import setup_logger
    setup_logger(logging.INFO)
    download_models()
