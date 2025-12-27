# config/settings.py
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data settings
TICKER = "AAPL"  # Stock symbol
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