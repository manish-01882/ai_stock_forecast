"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum


class TickerSymbol(BaseModel):
    """Ticker symbol validation."""
    ticker: str = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Stock ticker symbol (e.g., AAPL, GOOGL)"
    )
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Convert ticker to uppercase and strip whitespace."""
        return v.upper().strip()


class PredictionRequest(BaseModel):
    """Request model for stock price predictions."""
    ticker: str = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Stock ticker symbol",
        examples=["AAPL", "GOOGL", "MSFT"]
    )
    days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Number of days to predict (1-90)"
    )
    include_history: bool = Field(
        default=False,
        description="Include historical data in response"
    )
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Convert ticker to uppercase."""
        return v.upper().strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "days": 7,
                    "include_history": False
                }
            ]
        }
    }


class PredictionPoint(BaseModel):
    """Single prediction data point."""
    date: str = Field(..., description="Prediction date (YYYY-MM-DD)")
    price: float = Field(..., description="Predicted price in USD", ge=0)
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "date": "2026-01-15",
                    "price": 150.25
                }
            ]
        }
    }


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mse: float = Field(..., description="Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rmse": 2.15,
                    "mae": 1.75,
                    "mse": 4.62,
                    "mape": 1.8
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    ticker: str = Field(..., description="Stock ticker symbol")
    current_price: Optional[float] = Field(None, description="Current/latest price in USD")
    prediction_date: str = Field(..., description="Date when prediction was made")
    predictions: List[PredictionPoint] = Field(..., description="List of predicted prices")
    model_metrics: Optional[ModelMetrics] = Field(None, description="Model performance metrics")
    confidence_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1,
        description="Prediction confidence (0-1)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "current_price": 150.25,
                    "prediction_date": "2026-01-14",
                    "predictions": [
                        {"date": "2026-01-15", "price": 151.30},
                        {"date": "2026-01-16", "price": 152.10}
                    ],
                    "model_metrics": {
                        "rmse": 2.15,
                        "mae": 1.75,
                        "mse": 4.62,
                        "mape": 1.8
                    },
                    "confidence_score": 0.89
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Model metadata and information."""
    ticker: str = Field(..., description="Stock ticker this model is trained for")
    model_name: str = Field(..., description="Model identifier")
    model_path: str = Field(..., description="Path to model file")
    training_date: str = Field(..., description="When the model was trained")
    metrics: ModelMetrics = Field(..., description="Model performance metrics")
    lookback_window: Optional[int] = Field(None, description="Number of historical days used")
    total_epochs: Optional[int] = Field(None, description="Training epochs completed")
    architecture: Optional[str] = Field(None, description="Model architecture description")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "model_name": "lstm_paper_AAPL_20260114_1200",
                    "model_path": "models/lstm_paper_AAPL_20260114_1200_best.h5",
                    "training_date": "2026-01-14T12:00:00",
                    "metrics": {
                        "rmse": 2.15,
                        "mae": 1.75,
                        "mse": 4.62,
                        "mape": 1.8
                    },
                    "lookback_window": 60,
                    "total_epochs": 100,
                    "architecture": "Stacked LSTM (4 layers, 96 units each)"
                }
            ]
        }
    }


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingRequest(BaseModel):
    """Request model for training a new model."""
    ticker: str = Field(
        ..., 
        min_length=1, 
        max_length=10,
        description="Stock ticker to train model for"
    )
    epochs: Optional[int] = Field(
        None,
        ge=10,
        le=500,
        description="Number of training epochs (default from config)"
    )
    batch_size: Optional[int] = Field(
        None,
        ge=8,
        le=256,
        description="Training batch size (default from config)"
    )
    
    @field_validator('ticker')
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Convert ticker to uppercase."""
        return v.upper().strip()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ticker": "AAPL",
                    "epochs": 100,
                    "batch_size": 32
                }
            ]
        }
    }


class TrainingResponse(BaseModel):
    """Response for training job."""
    job_id: str = Field(..., description="Unique job identifier")
    ticker: str = Field(..., description="Stock ticker")
    status: TrainingStatus = Field(..., description="Current training status")
    message: str = Field(..., description="Status message")
    estimated_duration: Optional[str] = Field(None, description="Estimated training time")
    started_at: Optional[str] = Field(None, description="Job start time")
    completed_at: Optional[str] = Field(None, description="Job completion time")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "train_AAPL_20260114_120500",
                    "ticker": "AAPL",
                    "status": "running",
                    "message": "Training in progress...",
                    "estimated_duration": "5-10 minutes",
                    "started_at": "2026-01-14T12:05:00"
                }
            ]
        }
    }


class TrainingJobStatus(BaseModel):
    """Detailed training job status."""
    job_id: str = Field(..., description="Job identifier")
    ticker: str = Field(..., description="Stock ticker")
    status: TrainingStatus = Field(..., description="Job status")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Progress percentage")
    current_epoch: Optional[int] = Field(None, description="Current training epoch")
    total_epochs: Optional[int] = Field(None, description="Total epochs to train")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Current metrics")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "train_AAPL_20260114_120500",
                    "ticker": "AAPL",
                    "status": "running",
                    "progress": 65.5,
                    "current_epoch": 65,
                    "total_epochs": 100,
                    "metrics": {
                        "loss": 0.0012,
                        "val_loss": 0.0015
                    }
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Current server time")
    version: str = Field(..., description="API version")
    models_loaded: int = Field(..., description="Number of models currently loaded")
    available_models: List[str] = Field(..., description="Available ticker models")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "timestamp": "2026-01-14T12:00:00",
                    "version": "1.0.0",
                    "models_loaded": 8,
                    "available_models": ["AAPL", "GOOGL", "MSFT", "TSLA"]
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "ModelNotFound",
                    "message": "No trained model found for ticker XYZ",
                    "details": {"ticker": "XYZ"},
                    "timestamp": "2026-01-14T12:00:00"
                }
            ]
        }
    }


class ModelsListResponse(BaseModel):
    """Response for listing all available models."""
    total_models: int = Field(..., description="Total number of available models")
    models: List[ModelInfo] = Field(..., description="List of model information")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_models": 3,
                    "models": [
                        {
                            "ticker": "AAPL",
                            "model_name": "lstm_paper_AAPL_20260114",
                            "model_path": "models/lstm_paper_AAPL.h5",
                            "training_date": "2026-01-14",
                            "metrics": {"rmse": 2.15, "mae": 1.75, "mse": 4.62, "mape": 1.8}
                        }
                    ]
                }
            ]
        }
    }
