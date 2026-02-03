"""
Stock Forecast API - Main Application
Production-ready REST API for stock price predictions using LSTM models.
"""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from typing import Dict
import uuid
import os

from api.models import (
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    ModelsListResponse,
    TrainingRequest,
    TrainingResponse,
    TrainingJobStatus,
    TrainingStatus,
    HealthResponse,
    ErrorResponse,
    ModelMetrics,
    PredictionPoint
)
from api.utils import (
    make_predictions,
    list_available_models,
    load_model_for_ticker,
    validate_ticker,
    calculate_confidence_score,
    ModelNotFoundError,
    PredictionError,
    get_latest_model_path,
    get_model_metadata
)
from api import __version__
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training jobs tracking (in-memory for now, can be upgraded to Redis/DB)
training_jobs: Dict[str, TrainingJobStatus] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Stock Forecast API Starting...")
    logger.info(f"Version: {__version__}")
    logger.info(f"Models Directory: {settings.MODELS_DIR}")
    logger.info("=" * 60)
    
    # Check if models directory exists
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.DATA_RAW_DIR, exist_ok=True)
    
    # Log available models
    try:
        models = list_available_models()
        logger.info(f"Found {len(models)} trained models")
        for model in models:
            logger.info(f"  - {model['ticker']}: {model['model_name']}")
    except Exception as e:
        logger.warning(f"Could not list models: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Stock Forecast API Shutting Down...")


# Create FastAPI app
app = FastAPI(
    title="Stock Forecast API",
    description="""
    Production-ready REST API for stock price predictions using LSTM neural networks.
    
    ## Features
    - 📈 Generate multi-day price predictions for any ticker
    - 🤖 View trained model information and metrics
    - 🔄 Train new models on demand
    - ✅ Health monitoring
    
    ## Quick Start
    1. Check available models: `GET /api/v1/models`
    2. Get predictions: `POST /api/v1/predict` with ticker and days
    3. View interactive docs: `/docs`
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", response_class=HTMLResponse, tags=["Root"])
async def root():
    """API landing page with documentation links."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Forecast API</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            .version {{
                color: #999;
                font-size: 0.9em;
                margin-bottom: 30px;
            }}
            .endpoints {{
                margin: 30px 0;
            }}
            .endpoint {{
                background: #f8f9fa;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #667eea;
            }}
            .method {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
                margin-right: 10px;
                font-size: 0.85em;
            }}
            .post {{ background: #49cc90; color: white; }}
            .get {{ background: #61affe; color: white; }}
            a {{
                color: #667eea;
                text-decoration: none;
                font-weight: 500;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            .docs-link {{
                display: inline-block;
                background: #667eea;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                text-decoration: none;
                margin: 10px 10px 10px 0;
                transition: background 0.3s;
            }}
            .docs-link:hover {{
                background: #764ba2;
                text-decoration: none;
            }}
            code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📈 Stock Forecast API</h1>
            <div class="version">Version {__version__}</div>
            
            <p>Welcome to the Stock Forecast API! This service provides AI-powered stock price predictions using LSTM neural networks.</p>
            
            <h2>📚 Documentation</h2>
            <a href="/docs" class="docs-link">Interactive API Docs (Swagger UI)</a>
            <a href="/redoc" class="docs-link">ReDoc Documentation</a>
            
            <h2>🔗 Quick Reference</h2>
            <div class="endpoints">
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/health</code> - Health check
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/v1/models</code> - List all available models
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/v1/models/{{ticker}}</code> - Get specific model info
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/api/v1/predict</code> - Generate price predictions
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/api/v1/train</code> - Train a new model
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/api/v1/train/{{job_id}}</code> - Check training status
                </div>
            </div>
            
            <h2>💡 Example Request</h2>
            <pre><code>curl -X POST "http://localhost:8000/api/v1/predict" \\
  -H "Content-Type: application/json" \\
  -d '{{"ticker": "AAPL", "days": 7}}'</code></pre>
            
            <p style="margin-top: 30px; color: #999; font-size: 0.9em;">
                Built with FastAPI • TensorFlow • LSTM Neural Networks
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health Check",
    description="Check if the API is running and get system status"
)
async def health_check():
    """Health check endpoint."""
    try:
        models = list_available_models()
        available_tickers = [m['ticker'] for m in models]
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version=__version__,
            models_loaded=len(models),
            available_models=available_tickers
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "version": __version__,
                "error": str(e)
            }
        )


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Generate Price Predictions",
    description="Generate stock price predictions for the specified ticker and date range",
    responses={
        200: {"description": "Predictions generated successfully"},
        404: {"model": ErrorResponse, "description": "Model not found for ticker"},
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict(request: PredictionRequest):
    """Generate stock price predictions."""
    try:
        ticker = request.ticker.upper()
        
        # Validate ticker format
        if not validate_ticker(ticker):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ticker format: {ticker}"
            )
        
        logger.info(f"Prediction request: {ticker} for {request.days} days")
        
        # Generate predictions
        predictions_list, current_price, metadata = make_predictions(
            ticker=ticker,
            days=request.days,
            lookback=settings.LOOKBACK_WINDOW
        )
        
        # Extract metrics
        metrics = metadata.get('metrics', {})
        model_metrics = ModelMetrics(**metrics) if metrics else None
        
        # Calculate confidence score
        confidence = calculate_confidence_score(metadata)
        
        # Build response
        response = PredictionResponse(
            ticker=ticker,
            current_price=current_price,
            prediction_date=datetime.now().strftime('%Y-%m-%d'),
            predictions=[PredictionPoint(**p) for p in predictions_list],
            model_metrics=model_metrics,
            confidence_score=confidence
        )
        
        logger.info(f"Successfully generated {len(predictions_list)} predictions for {ticker}")
        return response
        
    except ModelNotFoundError as e:
        logger.warning(f"Model not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================================================

@app.get(
    "/api/v1/models",
    response_model=ModelsListResponse,
    tags=["Models"],
    summary="List All Models",
    description="Get a list of all available trained models with their metrics"
)
async def list_models():
    """List all available trained models."""
    try:
        models = list_available_models()
        
        # Convert to ModelInfo objects
        model_infos = []
        for model_data in models:
            metrics = model_data.get('metrics', {})
            model_info = ModelInfo(
                ticker=model_data['ticker'],
                model_name=model_data['model_name'],
                model_path=model_data['model_path'],
                training_date=model_data['training_date'],
                metrics=ModelMetrics(**metrics) if metrics else ModelMetrics(
                    rmse=0, mae=0, mse=0, mape=0
                ),
                lookback_window=model_data.get('lookback_window'),
                total_epochs=model_data.get('total_epochs'),
                architecture="Stacked LSTM (4 layers, 96 units each)"
            )
            model_infos.append(model_info)
        
        return ModelsListResponse(
            total_models=len(model_infos),
            models=model_infos
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.get(
    "/api/v1/models/{ticker}",
    response_model=ModelInfo,
    tags=["Models"],
    summary="Get Model Info",
    description="Get detailed information about a specific model",
    responses={
        404: {"model": ErrorResponse, "description": "Model not found"}
    }
)
async def get_model_info(ticker: str):
    """Get information about a specific model."""
    try:
        ticker = ticker.upper()
        
        # Get model path
        model_path = get_latest_model_path(ticker)
        
        if not model_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No model found for ticker {ticker}"
            )
        
        # Get metadata
        metadata = get_model_metadata(model_path) or {}
        
        metrics = metadata.get('metrics', {})
        
        model_info = ModelInfo(
            ticker=ticker,
            model_name=os.path.basename(model_path).replace('.h5', ''),
            model_path=model_path,
            training_date=metadata.get('training_date', 'unknown'),
            metrics=ModelMetrics(**metrics) if metrics else ModelMetrics(
                rmse=0, mae=0, mse=0, mape=0
            ),
            lookback_window=metadata.get('lookback_window', settings.LOOKBACK_WINDOW),
            total_epochs=metadata.get('total_epochs', settings.EPOCHS),
            architecture="Stacked LSTM (4 layers, 96 units each)"
        )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

def run_training_job(job_id: str, ticker: str, epochs: int, batch_size: int):
    """Background task to run model training."""
    try:
        # Update status to running
        training_jobs[job_id].status = TrainingStatus.RUNNING
        training_jobs[job_id].progress = 0
        
        logger.info(f"Starting training job {job_id} for {ticker}")
        
        # Import training modules
        from src.data_ingestor import DataIngestor
        from src.feature_engineer import FeatureEngineerUnivariate
        from src.model_trainer import ModelTrainer
        
        # Step 1: Data Ingestion
        training_jobs[job_id].progress = 10
        ingestor = DataIngestor(ticker=ticker)
        raw_file_path = ingestor.fetch_and_save()
        
        # Step 2: Feature Engineering
        training_jobs[job_id].progress = 30
        feature_engineer = FeatureEngineerUnivariate(
            lookback=settings.LOOKBACK_WINDOW,
            price_column='Close'
        )
        X_train, X_test, y_train, y_test, scaler = feature_engineer.process(raw_file_path)
        
        # Create validation split
        val_split = int(len(X_train) * 0.8)
        X_train_final = X_train[:val_split]
        y_train_final = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        
        # Step 3: Model Training
        training_jobs[job_id].progress = 50
        model_name = f"lstm_paper_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        trainer = ModelTrainer(model_name=model_name, ticker=ticker)
        
        model, history = trainer.train(
            X_train_final, y_train_final,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Step 4: Evaluation
        training_jobs[job_id].progress = 90
        metrics, y_test_actual, y_pred_actual = trainer.evaluate(
            model, X_test, y_test, scaler
        )
        
        # Save metadata
        training_date = datetime.now().isoformat()
        trainer.save_metadata(metrics, training_date=training_date)
        
        # Update job status
        training_jobs[job_id].status = TrainingStatus.COMPLETED
        training_jobs[job_id].progress = 100
        training_jobs[job_id].metrics = metrics
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
        training_jobs[job_id].status = TrainingStatus.FAILED
        training_jobs[job_id].error = str(e)


@app.post(
    "/api/v1/train",
    response_model=TrainingResponse,
    tags=["Training"],
    summary="Train New Model",
    description="Start a background job to train a new model for the specified ticker",
    status_code=status.HTTP_202_ACCEPTED
)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training a new model."""
    try:
        ticker = request.ticker.upper()
        
        # Validate ticker
        if not validate_ticker(ticker):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ticker format: {ticker}"
            )
        
        # Generate job ID
        job_id = f"train_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get training parameters
        epochs = request.epochs or settings.EPOCHS
        batch_size = request.batch_size or settings.BATCH_SIZE
        
        # Create job status
        job_status = TrainingJobStatus(
            job_id=job_id,
            ticker=ticker,
            status=TrainingStatus.PENDING,
            progress=0,
            current_epoch=0,
            total_epochs=epochs
        )
        training_jobs[job_id] = job_status
        
        # Start background training
        background_tasks.add_task(
            run_training_job,
            job_id=job_id,
            ticker=ticker,
            epochs=epochs,
            batch_size=batch_size
        )
        
        logger.info(f"Training job {job_id} queued for {ticker}")
        
        return TrainingResponse(
            job_id=job_id,
            ticker=ticker,
            status=TrainingStatus.PENDING,
            message=f"Training job started for {ticker}",
            estimated_duration="5-10 minutes",
            started_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training: {str(e)}"
        )


@app.get(
    "/api/v1/train/{job_id}",
    response_model=TrainingJobStatus,
    tags=["Training"],
    summary="Get Training Status",
    description="Check the status of a training job",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"}
    }
)
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    return training_jobs[job_id]


# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
