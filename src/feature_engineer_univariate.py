# src/feature_engineer_univariate.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineerUnivariate:
    """
    Univariate feature engineer based on research paper recommendations.
    Uses ONLY price data (Open or Close) without technical indicators.
    Paper 2 found that univariate models were more accurate and efficient.
    """
    def __init__(self, lookback=settings.LOOKBACK_WINDOW, price_column='Close'):
        self.lookback = lookback
        self.price_column = price_column  # 'Close' or 'Open'
        # MinMaxScaler works better for LSTM price predictions
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_sequences(self, data):
        """Creates time-series sequences for LSTM.
        
        Args:
            data: Normalized price data (1D array after scaling)
            
        Returns:
            X: Sequences of shape (samples, lookback, 1)
            y: Target values (next price)
        """
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])  # Past 'lookback' prices
            y.append(data[i, 0])  # Next price as target
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to add feature dimension: (samples, lookback, features=1)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y

    def process(self, input_path):
        """Main method to load raw data and create univariate sequences.
        
        Following Paper 1 and Paper 2 methodology:
        - Uses only price data (no technical indicators)
        - 80/20 train/test split
        - MinMaxScaler normalization
        - Creates sequences with 50-day lookback
        
        Args:
            input_path: Path to raw CSV data
            
        Returns:
            X_train, X_test, y_train, y_test, scaler
        """
        logger.info(f"Processing univariate data from {input_path}")

        # 1. Load raw data
        df = pd.read_csv(input_path, index_col='Date', parse_dates=True)
        
        # 2. Extract only the target price column
        if self.price_column not in df.columns:
            raise ValueError(f"Column '{self.price_column}' not found. Available: {df.columns.tolist()}")
        
        df_price = df[[self.price_column]].copy()
        logger.info(f"Loaded {len(df_price)} rows of {self.price_column} prices")
        
        # Check for missing values
        if df_price[self.price_column].isna().any():
            logger.warning(f"Found {df_price[self.price_column].isna().sum()} NaN values. Dropping them.")
            df_price = df_price.dropna()
        
        if len(df_price) < self.lookback + 100:
            raise ValueError(
                f"Insufficient data. Need at least {self.lookback + 100} rows, got {len(df_price)}. "
                f"Try adjusting START_DATE in settings.py to get more historical data."
            )

        # 3. Split into train and test BEFORE scaling (to prevent data leakage)
        split_idx = int(len(df_price) * settings.TRAIN_TEST_SPLIT)
        df_train = df_price.iloc[:split_idx]
        df_test = df_price.iloc[split_idx:]
        
        logger.info(f"Train data: {len(df_train)} rows, Test data: {len(df_test)} rows")

        # 4. Fit scaler on training data only (critical for preventing data leakage)
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
            raise ValueError(
                f"Not enough training data after lookback window. "
                f"Need more than {self.lookback} rows in training set. "
                f"Current training rows: {len(df_train)}"
            )
        
        X_train, X_test = X[:train_sequences], X[train_sequences:]
        y_train, y_test = y[:train_sequences], y[train_sequences:]

        logger.info(f"✓ Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
        logger.info(f"✓ Univariate model: 1 feature ('{self.price_column}' price only)")
        logger.info(f"✓ Lookback window: {self.lookback} days")
        
        return X_train, X_test, y_train, y_test, self.scaler


if __name__ == "__main__":
    # Test the univariate feature engineer
    import os
    ticker = settings.TICKER
    raw_file = os.path.join(settings.DATA_RAW_DIR, f"{ticker}_raw.csv")
    
    if os.path.exists(raw_file):
        fe = FeatureEngineerUnivariate()
        X_train, X_test, y_train, y_test, scaler = fe.process(raw_file)
        print(f"✓ Univariate processing successful!")
        print(f"  Training shape: {X_train.shape}")
        print(f"  Test shape: {X_test.shape}")
    else:
        print(f"✗ Raw data file not found: {raw_file}")
        print("  Run: python src/data_ingestor.py")
