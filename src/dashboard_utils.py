# src/dashboard_utils.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def load_trained_model(model_path):
    """Load a trained LSTM model with caching."""
    try:
        if os.path.exists(model_path):
            model = load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        else:
            logger.warning(f"Model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(ticker):
    """Load stock data from CSV and fetch latest live data from Yahoo Finance."""
    try:
        import yfinance as yf
        
        # Load historical data from CSV if available
        file_path = os.path.join(settings.DATA_RAW_DIR, f"{ticker}_raw.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            # Keep only main columns
            main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            df = df[[col for col in main_cols if col in df.columns]]
            last_csv_date = df.index.max()
            logger.info(f"Loaded {len(df)} rows from CSV, last date: {last_csv_date.date()}")
        else:
            df = None
            last_csv_date = None
            logger.warning(f"Data file not found: {file_path}, fetching all data from API")
        
        # Fetch live/recent data from Yahoo Finance
        # Always fetch at least last 30 days to ensure we have current data
        if last_csv_date:
            # Fetch from last CSV date to today to get any new data
            start_date = last_csv_date
        else:
            # If no CSV, fetch from settings start date
            start_date = settings.START_DATE
        
        logger.info(f"Fetching live data for {ticker} from {start_date}")
        live_data = yf.download(ticker, start=start_date, progress=False)
        
        if not live_data.empty:
            # Flatten multi-index columns if present
            if isinstance(live_data.columns, pd.MultiIndex):
                live_data.columns = live_data.columns.get_level_values(0)
            
            # Keep only main columns
            main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            live_data = live_data[[col for col in main_cols if col in live_data.columns]]
            
            # Merge with CSV data
            if df is not None:
                combined_df = pd.concat([df, live_data])
                # Remove duplicates, keeping the latest (live) data
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                logger.info(f"Combined data: {len(combined_df)} rows, new rows from API: {len(live_data)}")
            else:
                combined_df = live_data
                logger.info(f"Using live data only: {len(combined_df)} rows")
            
            return combined_df
        else:
            logger.warning("No live data available from Yahoo Finance")
            return df  # Return CSV data if available
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        # Try to return CSV data as fallback
        try:
            file_path = os.path.join(settings.DATA_RAW_DIR, f"{ticker}_raw.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
                df = df[[col for col in main_cols if col in df.columns]]
                logger.info(f"Fallback to CSV data: {len(df)} rows")
                return df
        except:
            pass
        return None


@st.cache_data
def add_technical_indicators(df):
    """Add technical indicators to dataframe."""
    df_copy = df.copy()
    
    # Add indicators
    df_copy.ta.sma(length=20, append=True)   # Simple Moving Average
    df_copy.ta.sma(length=50, append=True)   # 50-day SMA
    df_copy.ta.rsi(length=14, append=True)   # Relative Strength Index
    df_copy.ta.macd(append=True)             # MACD
    df_copy.ta.bbands(length=20, append=True) # Bollinger Bands
    df_copy['Returns'] = df_copy['Close'].pct_change()
    
    return df_copy


def prepare_data_for_prediction(df, lookback=100):
    """Prepare data for model prediction with comprehensive technical indicators."""
    # Add technical indicators (matching training pipeline)
    df_processed = df.copy()
    
    # Moving Averages
    df_processed.ta.sma(length=20, append=True)
    df_processed.ta.sma(length=50, append=True)
    df_processed.ta.ema(length=12, append=True)
    df_processed.ta.ema(length=26, append=True)
    
    # Momentum Indicators
    df_processed.ta.rsi(length=14, append=True)
    df_processed.ta.macd(append=True)
    df_processed.ta.stoch(append=True)
    df_processed.ta.adx(length=14, append=True)
    
    # Volatility Indicators
    df_processed.ta.bbands(length=20, append=True)
    df_processed.ta.atr(length=14, append=True)
    
    # Volume Indicators
    df_processed.ta.obv(append=True)
    df_processed.ta.ad(append=True)
    
    # Price-based features
    df_processed['Returns'] = df_processed['Close'].pct_change()
    df_processed['Log_Returns'] = np.log(df_processed['Close'] / df_processed['Close'].shift(1))
    
    # Volatility features
    df_processed['Volatility_5'] = df_processed['Returns'].rolling(window=5).std()
    df_processed['Volatility_20'] = df_processed['Returns'].rolling(window=20).std()
    
    # Price momentum features
    df_processed['Price_Change_5d'] = df_processed['Close'].pct_change(periods=5)
    df_processed['Price_Change_20d'] = df_processed['Close'].pct_change(periods=20)
    
    # Volume momentum
    df_processed['Volume_Change'] = df_processed['Volume'].pct_change()
    df_processed['Volume_MA_20'] = df_processed['Volume'].rolling(window=20).mean()
    df_processed['Volume_Ratio'] = df_processed['Volume'] / df_processed['Volume_MA_20']
    
    # High-Low spread
    df_processed['HL_Spread'] = (df_processed['High'] - df_processed['Low']) / df_processed['Close']
    
    # Distance from moving averages
    df_processed['Close_to_SMA20'] = (df_processed['Close'] - df_processed['SMA_20']) / df_processed['SMA_20']
    df_processed['Close_to_SMA50'] = (df_processed['Close'] - df_processed['SMA_50']) / df_processed['SMA_50']
    
    df_processed = df_processed.dropna()
    
    # Scale the data using RobustScaler
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_processed)
    
    # Create sequences
    X = []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, :])
    
    X = np.array(X)
    
    return X, scaler, df_processed


def make_predictions(model, df, lookback=100):
    """Generate predictions using the trained model."""
    if model is None or df is None or len(df) < lookback:
        return None, None, None
    
    X, scaler, df_processed = prepare_data_for_prediction(df, lookback)
    
    if len(X) == 0:
        return None, None, None
    
    # Make predictions
    predictions_scaled = model.predict(X)
    
    # Inverse transform predictions
    dummy_array = np.zeros((len(predictions_scaled), scaler.n_features_in_))
    dummy_array[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Get actual values for comparison
    actual_values = df_processed['Close'].values[lookback:]
    
    # Create dates for predictions
    prediction_dates = df_processed.index[lookback:]
    
    return predictions, actual_values, prediction_dates


def predict_future(model, df, days_ahead=30, lookback=100):
    """Predict future stock prices."""
    if model is None or df is None:
        return None, None
    
    # Prepare the most recent data with all technical indicators
    df_processed = df.copy()
    
    # Moving Averages
    df_processed.ta.sma(length=20, append=True)
    df_processed.ta.sma(length=50, append=True)
    df_processed.ta.ema(length=12, append=True)
    df_processed.ta.ema(length=26, append=True)
    
    # Momentum Indicators
    df_processed.ta.rsi(length=14, append=True)
    df_processed.ta.macd(append=True)
    df_processed.ta.stoch(append=True)
    df_processed.ta.adx(length=14, append=True)
    
    # Volatility Indicators
    df_processed.ta.bbands(length=20, append=True)
    df_processed.ta.atr(length=14, append=True)
    
    # Volume Indicators
    df_processed.ta.obv(append=True)
    df_processed.ta.ad(append=True)
    
    # Price-based features
    df_processed['Returns'] = df_processed['Close'].pct_change()
    df_processed['Log_Returns'] = np.log(df_processed['Close'] / df_processed['Close'].shift(1))
    
    # Volatility features
    df_processed['Volatility_5'] = df_processed['Returns'].rolling(window=5).std()
    df_processed['Volatility_20'] = df_processed['Returns'].rolling(window=20).std()
    
    # Price momentum features
    df_processed['Price_Change_5d'] = df_processed['Close'].pct_change(periods=5)
    df_processed['Price_Change_20d'] = df_processed['Close'].pct_change(periods=20)
    
    # Volume momentum
    df_processed['Volume_Change'] = df_processed['Volume'].pct_change()
    df_processed['Volume_MA_20'] = df_processed['Volume'].rolling(window=20).mean()
    df_processed['Volume_Ratio'] = df_processed['Volume'] / df_processed['Volume_MA_20']
    
    # High-Low spread
    df_processed['HL_Spread'] = (df_processed['High'] - df_processed['Low']) / df_processed['Close']
    
    # Distance from moving averages
    df_processed['Close_to_SMA20'] = (df_processed['Close'] - df_processed['SMA_20']) / df_processed['SMA_20']
    df_processed['Close_to_SMA50'] = (df_processed['Close'] - df_processed['SMA_50']) / df_processed['SMA_50']
    
    df_processed = df_processed.dropna()
    
    # Scale
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df_processed)
    
    # Get the last lookback window
    last_sequence = scaled_data[-lookback:]
    
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Reshape for prediction
        current_input = current_sequence.reshape(1, lookback, -1)
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)
        
        # Create full feature vector for the prediction
        next_features = np.zeros((1, scaler.n_features_in_))
        next_features[0, 0] = next_pred[0, 0]
        
        # For simplicity, use the last known values for other features
        # In a more sophisticated approach, you'd predict or estimate these too
        next_features[0, 1:] = current_sequence[-1, 1:]
        
        # Append to predictions
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence
        current_sequence = np.vstack([current_sequence[1:], next_features])
    
    # Inverse transform predictions
    dummy_array = np.zeros((len(future_predictions), scaler.n_features_in_))
    dummy_array[:, 0] = future_predictions
    future_prices = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Generate future dates
    last_date = df_processed.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
    
    return future_prices, future_dates


@st.cache_data
def calculate_metrics(actual, predicted):
    """Calculate performance metrics."""
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape
    }


def get_available_models():
    """Get list of available trained models."""
    models_dir = settings.MODELS_DIR
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    return sorted(model_files, reverse=True)  # Most recent first


def get_model_info(model_path):
    """Extract information from model filename."""
    filename = os.path.basename(model_path)
    name = filename.replace('.h5', '')
    
    # Try to extract date from filename
    try:
        if '_' in name:
            parts = name.split('_')
            date_part = parts[-1] if len(parts[-1]) >= 8 else parts[-2] if len(parts) > 1 else ''
            if len(date_part) >= 8:
                date_str = date_part[:8]
                model_date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            else:
                model_date = 'Unknown'
        else:
            model_date = 'Unknown'
    except:
        model_date = 'Unknown'
    
    return {
        'name': name,
        'date': model_date,
        'path': model_path
    }
