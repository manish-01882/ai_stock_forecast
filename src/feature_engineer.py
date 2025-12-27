# src/feature_engineer.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
import pandas_ta as ta  # Technical analysis library
from sklearn.preprocessing import MinMaxScaler
import os
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, lookback=settings.LOOKBACK_WINDOW):
        self.lookback = lookback
        # MinMaxScaler works better for LSTM price predictions
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_technical_indicators(self, df):
        """Adds carefully selected technical indicators to avoid overfitting."""
        logger.info("Adding technical indicators...")
        
        # Core Moving Averages (most important trend indicators)
        df.ta.sma(length=20, append=True)   # Short-term MA
        df.ta.sma(length=50, append=True)   # Long-term MA
        df.ta.ema(length=12, append=True)   # Fast EMA for MACD
        
        # Key Momentum Indicators
        df.ta.rsi(length=14, append=True)   # RSI - very important
        df.ta.macd(append=True)             # MACD - trend + momentum
        
        # Volatility Indicators
        df.ta.bbands(length=20, append=True) # Bollinger Bands
        df.ta.atr(length=14, append=True)    # Average True Range
        
        # Volume Indicator (only the most useful one)
        df.ta.obv(append=True)               # On-Balance Volume
        
        # Simple price-based features (avoid over-engineering)
        df['Returns'] = df['Close'].pct_change()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Volume ratio (simple but effective)
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # Drop rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"Data shape after adding indicators: {df.shape}")
        logger.info(f"Dropped {dropped_rows} rows due to NaN values")
        logger.info(f"Total features: {len(df.columns)}")
        
        return df

    def create_sequences(self, data):
        """Creates time-series sequences for LSTM."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, :])  # Past 'lookback' steps as features
            y.append(data[i, 0])  # Next 'Close' price as target
        return np.array(X), np.array(y)

    def process(self, input_path):
        """Main method to load raw data, add features, and create sequences."""
        logger.info(f"Processing data from {input_path}")

        # 1. Load raw data
        df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
        
        # Keep only main columns (defensive against duplicate columns)
        main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
        df = df[[col for col in main_cols if col in df.columns]]
        logger.info(f"Loaded data shape: {df.shape}")

        # 2. Add technical indicators
        df_processed = self.add_technical_indicators(df)
        
        if len(df_processed) < self.lookback + 100:
            raise ValueError(f"Insufficient data after feature engineering. Need at least {self.lookback + 100} rows, got {len(df_processed)}")

        # 3. Split into train and test BEFORE scaling (to prevent data leakage)
        split_idx = int(len(df_processed) * settings.TRAIN_TEST_SPLIT)
        df_train = df_processed.iloc[:split_idx]
        df_test = df_processed.iloc[split_idx:]
        
        logger.info(f"Train data: {len(df_train)} rows, Test data: {len(df_test)} rows")

        # 4. Fit scaler on training data only
        scaled_train = self.scaler.fit_transform(df_train)
        scaled_test = self.scaler.transform(df_test)
        
        # Combine back for sequence creation
        scaled_data = np.vstack([scaled_train, scaled_test])

        # 5. Create sequences for LSTM
        X, y = self.create_sequences(scaled_data)
        
        # The split needs to account for lookback window
        # After creating sequences, we have fewer samples
        train_sequences = len(scaled_train) - self.lookback
        
        if train_sequences <= 0:
            raise ValueError(f"Not enough training data after lookback. Need more than {self.lookback} rows in training set")
        
        X_train, X_test = X[:train_sequences], X[train_sequences:]
        y_train, y_test = y[:train_sequences], y[train_sequences:]

        logger.info(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
        logger.info(f"Feature dimensions: {X_train.shape[2]} features per timestep")
        
        return X_train, X_test, y_train, y_test, self.scaler