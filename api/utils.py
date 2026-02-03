"""
Utility functions for API operations.
Handles model loading, predictions, and data management.
"""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

from config import settings

logger = logging.getLogger(__name__)

# Global cache for loaded models and scalers
_MODEL_CACHE: Dict[str, Tuple[keras.Model, dict]] = {}


class ModelNotFoundError(Exception):
    """Raised when no model is found for a ticker."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


def get_latest_model_path(ticker: str) -> Optional[str]:
    """
    Find the latest trained model for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Path to the best model file, or None if not found
    """
    models_dir = settings.MODELS_DIR
    
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory does not exist: {models_dir}")
        return None
    
    # Look for model files matching this ticker
    # Priority: *_best.h5 > *.h5
    best_models = []
    regular_models = []
    
    for filename in os.listdir(models_dir):
        if ticker.upper() in filename.upper() and filename.endswith('.h5'):
            full_path = os.path.join(models_dir, filename)
            if '_best.h5' in filename:
                best_models.append(full_path)
            else:
                regular_models.append(full_path)
    
    # Get the most recent file
    if best_models:
        best_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return best_models[0]
    elif regular_models:
        regular_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return regular_models[0]
    
    return None


def get_model_metadata(model_path: str) -> Optional[Dict]:
    """
    Load metadata JSON file for a model.
    
    Args:
        model_path: Path to .h5 model file
        
    Returns:
        Metadata dictionary or None if not found
    """
    # Try to find corresponding .json file
    base_path = model_path.replace('_best.h5', '').replace('.h5', '')
    json_path = f"{base_path}.json"
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {json_path}: {e}")
    
    return None


def load_model_for_ticker(ticker: str, use_cache: bool = True) -> Tuple[keras.Model, Dict]:
    """
    Load a trained model and its metadata for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        use_cache: Whether to use cached model if available
        
    Returns:
        Tuple of (model, metadata_dict)
        
    Raises:
        ModelNotFoundError: If no model exists for the ticker
    """
    ticker = ticker.upper()
    
    # Check cache first
    if use_cache and ticker in _MODEL_CACHE:
        logger.info(f"Using cached model for {ticker}")
        return _MODEL_CACHE[ticker]
    
    # Find model file
    model_path = get_latest_model_path(ticker)
    
    if not model_path:
        raise ModelNotFoundError(f"No trained model found for ticker {ticker}")
    
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}")
        model = keras.models.load_model(model_path)
        
        # Load metadata
        metadata = get_model_metadata(model_path) or {}
        
        # Add model path to metadata
        metadata['model_path'] = model_path
        metadata['ticker'] = ticker
        
        # Cache the model
        _MODEL_CACHE[ticker] = (model, metadata)
        
        logger.info(f"Successfully loaded model for {ticker}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Failed to load model for {ticker}: {e}")
        raise PredictionError(f"Failed to load model: {str(e)}")


def get_latest_price_data(ticker: str, lookback: int = 60) -> Tuple[np.ndarray, MinMaxScaler, float]:
    """
    Get the latest price data for making predictions.
    
    Args:
        ticker: Stock ticker symbol
        lookback: Number of historical days needed
        
    Returns:
        Tuple of (scaled_sequence, scaler, latest_price)
        
    Raises:
        PredictionError: If data cannot be loaded
    """
    try:
        # Load raw data
        raw_file = os.path.join(settings.DATA_RAW_DIR, f"{ticker}_raw.csv")
        
        if not os.path.exists(raw_file):
            raise PredictionError(f"No data file found for {ticker}. Please ingest data first.")
        
        df = pd.read_csv(raw_file, index_col='Date', parse_dates=True)
        
        if 'Close' not in df.columns:
            raise PredictionError(f"'Close' column not found in data for {ticker}")
        
        # Get the last 'lookback' prices
        prices = df['Close'].values[-lookback:]
        
        if len(prices) < lookback:
            raise PredictionError(
                f"Insufficient data for {ticker}. Need {lookback} days, have {len(prices)}"
            )
        
        latest_price = float(prices[-1])
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_reshaped = prices.reshape(-1, 1)
        scaled_prices = scaler.fit_transform(prices_reshaped)
        
        # Reshape for LSTM: (1, lookback, 1)
        sequence = scaled_prices.reshape(1, lookback, 1)
        
        return sequence, scaler, latest_price
        
    except Exception as e:
        logger.error(f"Failed to get price data for {ticker}: {e}")
        raise PredictionError(f"Failed to load price data: {str(e)}")


def make_predictions(
    ticker: str, 
    days: int = 7,
    lookback: int = 60
) -> Tuple[List[Dict], float, Dict]:
    """
    Generate price predictions for the next N days.
    
    Args:
        ticker: Stock ticker symbol
        days: Number of days to predict
        lookback: Historical window size
        
    Returns:
        Tuple of (predictions_list, current_price, metadata)
        
    Raises:
        ModelNotFoundError: If no model found
        PredictionError: If prediction fails
    """
    ticker = ticker.upper()
    
    # Load model
    model, metadata = load_model_for_ticker(ticker)
    
    # Get lookback from metadata if available
    if 'lookback_window' in metadata:
        lookback = metadata['lookback_window']
    
    # Get latest price data
    last_sequence, scaler, current_price = get_latest_price_data(ticker, lookback)
    
    # Make predictions iteratively
    predictions = []
    current_sequence = last_sequence.copy()
    
    # Start date is tomorrow
    start_date = datetime.now() + timedelta(days=1)
    
    for i in range(days):
        # Predict next price
        pred_scaled = model.predict(current_sequence, verbose=0)
        
        # Inverse scale to get actual price
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        
        # Store prediction
        pred_date = start_date + timedelta(days=i)
        predictions.append({
            'date': pred_date.strftime('%Y-%m-%d'),
            'price': float(pred_price)
        })
        
        # Update sequence for next prediction
        # Remove first element, add new prediction
        new_sequence = np.append(current_sequence[0, 1:, :], pred_scaled.reshape(1, 1), axis=0)
        current_sequence = new_sequence.reshape(1, lookback, 1)
    
    logger.info(f"Generated {days} predictions for {ticker}")
    
    return predictions, current_price, metadata


def list_available_models() -> List[Dict]:
    """
    List all available trained models.
    
    Returns:
        List of dictionaries with model information
    """
    models_dir = settings.MODELS_DIR
    available_models = []
    
    if not os.path.exists(models_dir):
        return []
    
    # Find all unique tickers with models
    tickers_found = set()
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.h5'):
            # Extract ticker from filename
            # Pattern: lstm_paper_TICKER_datetime.h5
            parts = filename.split('_')
            for i, part in enumerate(parts):
                if part.isupper() and len(part) <= 10 and i > 0:
                    tickers_found.add(part)
                    break
    
    # Get info for each ticker
    for ticker in sorted(tickers_found):
        try:
            model_path = get_latest_model_path(ticker)
            if model_path:
                metadata = get_model_metadata(model_path) or {}
                
                model_info = {
                    'ticker': ticker,
                    'model_path': model_path,
                    'model_name': os.path.basename(model_path).replace('.h5', ''),
                    'training_date': metadata.get('training_date', 'unknown'),
                    'metrics': metadata.get('metrics', {}),
                    'lookback_window': metadata.get('lookback_window'),
                    'total_epochs': metadata.get('total_epochs'),
                }
                
                available_models.append(model_info)
        except Exception as e:
            logger.warning(f"Failed to get info for ticker {ticker}: {e}")
            continue
    
    return available_models


def validate_ticker(ticker: str) -> bool:
    """
    Basic validation for ticker symbols.
    
    Args:
        ticker: Stock ticker to validate
        
    Returns:
        True if valid format
    """
    if not ticker:
        return False
    
    ticker = ticker.strip().upper()
    
    # Basic checks
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    # Should contain only letters and possibly dots/dashes
    if not all(c.isalpha() or c in '.-' for c in ticker):
        return False
    
    return True


def calculate_confidence_score(metadata: Dict) -> float:
    """
    Calculate a confidence score based on model metrics.
    
    Args:
        metadata: Model metadata dictionary
        
    Returns:
        Confidence score between 0 and 1
    """
    if 'metrics' not in metadata:
        return 0.5  # Default moderate confidence
    
    metrics = metadata['metrics']
    
    # Lower MAPE = higher confidence
    # MAPE < 2% = excellent (0.9+)
    # MAPE 2-5% = good (0.7-0.9)
    # MAPE 5-10% = moderate (0.5-0.7)
    # MAPE > 10% = low (< 0.5)
    
    mape = metrics.get('mape', 10.0)
    
    if mape < 2:
        confidence = 0.95
    elif mape < 5:
        confidence = 0.9 - ((mape - 2) * 0.067)  # Linear scale from 0.9 to 0.7
    elif mape < 10:
        confidence = 0.7 - ((mape - 5) * 0.04)   # Linear scale from 0.7 to 0.5
    else:
        confidence = max(0.3, 0.5 - ((mape - 10) * 0.02))  # Max 0.3 minimum
    
    return round(confidence, 2)


def clear_model_cache():
    """Clear the model cache to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    logger.info("Model cache cleared")
