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
    """Prepare data for model prediction using UNIVARIATE approach (Close price only).
    
    This MUST match the training pipeline which uses only Close price without
    technical indicators, as per the research paper recommendations.
    """
    df_processed = df.copy()
    
    # Extract only Close price (univariate approach)
    price_data = df_processed[['Close']].copy()
    
    # Drop any NaN values
    price_data = price_data.dropna()
    
    # Scale the data using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_data)
    
    # Create sequences
    X = []
    for i in range(lookback, len(scaled_data)):
        # Shape: (lookback, 1) for each sample
        X.append(scaled_data[i-lookback:i, :])
    
    X = np.array(X)
    
    return X, scaler, price_data


def make_predictions(model, df, lookback=100):
    """Generate predictions using the trained model."""
    if model is None or df is None or len(df) < lookback:
        return None, None, None
    
    X, scaler, price_data = prepare_data_for_prediction(df, lookback)
    
    if len(X) == 0:
        return None, None, None
    
    # Make predictions
    predictions_scaled = model.predict(X)
    
    # Inverse transform predictions
    dummy_array = predictions_scaled.reshape(-1, 1)
    predictions = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Get actual values for comparison
    actual_values = price_data['Close'].values[lookback:]
    
    # Create dates for predictions
    prediction_dates = price_data.index[lookback:]
    
    return predictions, actual_values, prediction_dates


def predict_future(model, df, days_ahead=30, lookback=100):
    """Predict future stock prices using UNIVARIATE approach (Close price only).
    
    This MUST match the training pipeline which uses only Close price without
    technical indicators, as per the research paper recommendations.
    """
    if model is None or df is None:
        return None, None
    
    df_processed = df.copy()
    
    # Extract only Close price (univariate approach)
    price_data = df_processed[['Close']].copy()
    price_data = price_data.dropna()
    
    # Scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_data)
    
    # Get the last lookback window
    last_sequence = scaled_data[-lookback:]
    
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Reshape for prediction: (1, lookback, 1)
        current_input = current_sequence.reshape(1, lookback, 1)
        
        # Predict next value
        next_pred = model.predict(current_input, verbose=0)
        
        # Append to predictions
        future_predictions.append(next_pred[0, 0])
        
        # Update sequence by removing oldest and adding newest prediction
        # Shape remains (lookback, 1)
        current_sequence = np.vstack([current_sequence[1:], [[next_pred[0, 0]]]])
    
    # Inverse transform predictions
    dummy_array = np.array(future_predictions).reshape(-1, 1)
    future_prices = scaler.inverse_transform(dummy_array)[:, 0]
    
    # Generate future dates
    last_date = price_data.index[-1]
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


def get_model_for_ticker(ticker):
    """Get the most recent trained model for a specific ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        dict: Model info with path and metadata, or None if not found
    """
    models_dir = settings.MODELS_DIR
    if not os.path.exists(models_dir):
        return None
    
    # Find all metadata JSON files for this ticker
    pattern = f"lstm_paper_{ticker}_"
    metadata_files = [
        f for f in os.listdir(models_dir) 
        if f.startswith(pattern) and f.endswith('.json')
    ]
    
    if not metadata_files:
        return None
    
    # Get the most recent metadata file (sorted by filename/timestamp)
    metadata_files.sort(reverse=True)
    metadata_path = os.path.join(models_dir, metadata_files[0])
    
    # Load metadata
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        model_path = metadata.get('model_path')
        
        # Verify model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        return {
            'path': model_path,
            'metadata': metadata
        }
    except Exception as e:
        logger.error(f"Error loading metadata for {ticker}: {e}")
        return None


def get_available_models():
    """Get list of available trained models (legacy function for backward compatibility)."""
    models_dir = settings.MODELS_DIR
    if not os.path.exists(models_dir):
        return []
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
    return sorted(model_files, reverse=True)


def get_model_info(model_path):
    """Extract information from model (tries JSON metadata first, falls back to filename).
    
    Args:
        model_path: Path to model file
        
    Returns:
        dict: Model information
    """
    filename = os.path.basename(model_path)
    name = filename.replace('.h5', '')
    
    # Try to load metadata from JSON
    json_path = model_path.replace('.h5', '.json')
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            return {
                'name': name,
                'ticker': metadata.get('ticker', 'Unknown'),
                'date': metadata.get('training_date', 'Unknown')[:10],  # Just the date part
                'path': model_path,
                'metrics': metadata.get('metrics', {}),
                'metadata': metadata
            }
        except Exception as e:
            logger.warning(f"Could not load metadata from {json_path}: {e}")
    
    # Fallback: Try to extract from filename
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
        'ticker': 'Unknown',
        'date': model_date,
        'path': model_path,
        'metrics': {},
        'metadata': None
    }

