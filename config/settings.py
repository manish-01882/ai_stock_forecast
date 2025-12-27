# config/settings.py
import os
from datetime import datetime
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load tickers from config.yaml
config_path = os.path.join(BASE_DIR, 'config', 'config.yaml')
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        TICKERS = config.get('tickers', ["AAPL", "GOOGL", "MSFT"])
else:
    TICKERS = ["AAPL", "GOOGL", "MSFT"]

# Data settings
TICKER = "AAPL"  # Default stock symbol (for backward compatibility)
START_DATE = "2020-01-01"

# Feature Engineering (Based on Research Papers)
LOOKBACK_WINDOW = 50  # 50-day window as per Paper 1
TRAIN_TEST_SPLIT = 0.80  # 80/20 split as per Paper 1

# Model Configuration
USE_UNIVARIATE = True  # True: Use only price data, False: Use technical indicators
EPOCHS = 100  # Training epochs (Paper 1 recommends 50-100)
BATCH_SIZE = 32  # Batch size for training
USE_EARLY_STOPPING = False  # Set to True to enable early stopping

# Paths
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Helper functions
def get_model_path_for_ticker(ticker, timestamp=None):
    """Generate model path for a specific ticker."""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    return os.path.join(MODELS_DIR, f"lstm_paper_{ticker}_{timestamp}.h5")

def get_latest_model_for_ticker(ticker):
    """Get the most recent model file for a specific ticker."""
    if not os.path.exists(MODELS_DIR):
        return None
    
    # Find all models for this ticker
    pattern = f"lstm_paper_{ticker}_"
    matching_models = [
        f for f in os.listdir(MODELS_DIR) 
        if f.startswith(pattern) and f.endswith('.h5') and not f.endswith('_best.h5')
    ]
    
    if not matching_models:
        return None
    
    # Return the most recent (sorted by filename which includes timestamp)
    matching_models.sort(reverse=True)
    return os.path.join(MODELS_DIR, matching_models[0])