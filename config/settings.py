# config/settings.py
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data settings
TICKER = "AAPL"  # Stock symbol
START_DATE = "2020-01-01"

# Feature Engineering
LOOKBACK_WINDOW = 100  # For creating LSTM sequences[citation:9]
TRAIN_TEST_SPLIT = 0.65  # 65% for training[citation:9]

# Paths
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)