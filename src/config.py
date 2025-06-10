"""
Configuration settings for the AI Test Script Generator
"""

import os
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
EXAMPLES_DIR = ROOT_DIR / "examples"

# Model settings
DEFAULT_MODEL_NAME = "test-script-generator-small"
MODEL_CONFIG = {
    "test-script-generator-small": {
        "model_type": "sentence-transformer",
        "model_name": "paraphrase-MiniLM-L3-v2",  # Small model suitable for low-spec machines
        "embedding_dim": 384,
        "max_seq_length": 256,
    },
    "test-script-generator-medium": {
        "model_type": "sentence-transformer",
        "model_name": "paraphrase-MiniLM-L6-v2",  # Medium model with better performance
        "embedding_dim": 384,
        "max_seq_length": 384,
    },
}

# SWTBot template settings
TEMPLATE_DIR = SRC_DIR / "templates"
DEFAULT_TEMPLATE = "swtbot_template.java.j2"

# Logging settings
LOG_LEVEL = "INFO"

# Application settings
MAX_BATCH_SIZE = 8  # For low-spec machines
DEVICE = "cpu"  # Default to CPU for low-spec machines

# Flask web application settings
class Config:
    """Flask application configuration"""
    HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    
    # Model paths
    GTX1660_MODEL_PATH = os.environ.get('GTX1660_MODEL_PATH', 
                                      str(ROOT_DIR / '..' / 'models' / 'swtbot-gtx1660-optimized'))
    FALLBACK_MODEL_PATH = os.environ.get('FALLBACK_MODEL_PATH',
                                       str(ROOT_DIR / '..' / 'models' / 'swtbot-fine-tuned'))

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, EXAMPLES_DIR, TEMPLATE_DIR]:
    os.makedirs(directory, exist_ok=True)
