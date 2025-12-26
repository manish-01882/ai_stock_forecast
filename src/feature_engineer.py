# src/feature_engineer.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
import pandas_ta as ta  # Technical analysis library
from sklearn.preprocessing import RobustScaler
import os
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, lookback=settings.LOOKBACK_WINDOW):
        self.lookback = lookback
        # RobustScaler is better for financial data as it handles outliers
        self.scaler = RobustScaler()

    def add_technical_indicators(self, df):
        """Adds comprehensive technical indicators to the dataframe using pandas_ta."""
        logger.info("Adding technical indicators...")
        
        # Moving Averages
        df.ta.sma(length=20, append=True)   # Simple Moving Average 20
        df.ta.sma(length=50, append=True)   # Simple Moving Average 50
        df.ta.ema(length=12, append=True)   # Exponential Moving Average 12
        df.ta.ema(length=26, append=True)   # Exponential Moving Average 26
        
        # Momentum Indicators
        df.ta.rsi(length=14, append=True)   # Relative Strength Index
        df.ta.macd(append=True)             # MACD
        df.ta.stoch(append=True)            # Stochastic Oscillator
        df.ta.adx(length=14, append=True)   # Average Directional Index (trend strength)
        
        # Volatility Indicators
        df.ta.bbands(length=20, append=True) # Bollinger Bands
        df.ta.atr(length=14, append=True)    # Average True Range
        
        # Volume Indicators
        df.ta.obv(append=True)               # On-Balance Volume
        df.ta.ad(append=True)                # Accumulation/Distribution
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Price momentum features
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        df['Price_Change_20d'] = df['Close'].pct_change(periods=20)
        
        # Volume momentum
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # High-Low spread (intraday volatility)
        df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Distance from moving averages
        df['Close_to_SMA20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        df['Close_to_SMA50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        
        # Drop rows with NaN values (from indicator calculations)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"Data shape after adding indicators: {df.shape}")
        logger.info(f"Dropped {dropped_rows} rows due to NaN values from indicator calculations")
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