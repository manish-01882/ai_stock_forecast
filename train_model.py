#!/usr/bin/env python3
"""
train_model.py

End-to-end training script for the research paper-based LSTM model.
Uses univariate approach (price-only) with Stacked LSTM architecture.

Based on:
- Paper 1: Moghar & Hamiche (2020) - Stacked LSTM architecture
- Paper 2: Sen & Mehtab (2020) - Univariate superiority finding
"""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging

from src.data_ingestor import DataIngestor
from src.feature_engineer import FeatureEngineerUnivariate
from src.model_trainer import ModelTrainer
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_results(history, y_test_actual, y_pred_actual, metrics, save_path):
    """Creates visualization of training results and predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Model Training Results (Paper-Based Architecture)', fontsize=16, fontweight='bold')
    
    # Plot 1: Training & Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    ax2 = axes[0, 1]
    ax2.plot(y_test_actual, label='Actual Price', linewidth=2, alpha=0.7)
    ax2.plot(y_pred_actual, label='Predicted Price', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Price ($)')
    ax2.set_title('Predictions vs Actual Prices')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction Error Distribution
    ax3 = axes[1, 0]
    errors = y_test_actual - y_pred_actual
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    metrics_text = f"""
    Model Performance Metrics
    {'='*40}
    
    RMSE:  ${metrics['rmse']:.2f}
    MAE:   ${metrics['mae']:.2f}
    MAPE:  {metrics['mape']:.2f}%
    MSE:   ${metrics['mse']:.2f}
    
    {'='*40}
    Architecture: 4 Stacked LSTM layers
    Units per layer: 96
    Dropout rate: 0.2
    Lookback window: {settings.LOOKBACK_WINDOW} days
    Train/Test split: {int(settings.TRAIN_TEST_SPLIT*100)}/{int((1-settings.TRAIN_TEST_SPLIT)*100)}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Results plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("STOCK FORECAST - RESEARCH PAPER-BASED LSTM MODEL")
    logger.info("=" * 80)
    logger.info(f"Ticker: {settings.TICKER}")
    logger.info(f"Lookback Window: {settings.LOOKBACK_WINDOW} days")
    logger.info(f"Train/Test Split: {settings.TRAIN_TEST_SPLIT:.0%}/{1-settings.TRAIN_TEST_SPLIT:.0%}")
    logger.info(f"Epochs: {settings.EPOCHS}")
    logger.info(f"Batch Size: {settings.BATCH_SIZE}")
    logger.info(f"Early Stopping: {'Enabled' if settings.USE_EARLY_STOPPING else 'Disabled'}")
    logger.info("=" * 80)
    
    # Step 1: Data Ingestion
    logger.info("\n[STEP 1/4] Fetching stock data...")
    ingestor = DataIngestor(ticker=settings.TICKER)
    raw_file_path = ingestor.fetch_and_save()
    
    # Step 2: Feature Engineering (Univariate)
    logger.info("\n[STEP 2/4] Processing data (univariate approach)...")
    feature_engineer = FeatureEngineerUnivariate(
        lookback=settings.LOOKBACK_WINDOW,
        price_column='Close'  # Using Close price as per common practice
    )
    X_train, X_test, y_train, y_test, scaler = feature_engineer.process(raw_file_path)
    
    # Create validation set from training data (80/20 split of training data)
    val_split = int(len(X_train) * 0.8)
    X_train_final = X_train[:val_split]
    y_train_final = y_train[:val_split]
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    
    logger.info(f"  Final training set: {X_train_final.shape}")
    logger.info(f"  Validation set: {X_val.shape}")
    logger.info(f"  Test set: {X_test.shape}")
    
    # Step 3: Model Training
    logger.info("\n[STEP 3/4] Training LSTM model...")
    model_name = f"lstm_paper_{settings.TICKER}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    trainer = ModelTrainer(model_name=model_name, ticker=settings.TICKER)
    
    model, history = trainer.train(
        X_train_final, y_train_final,
        X_val, y_val,
        epochs=settings.EPOCHS,
        batch_size=settings.BATCH_SIZE
    )
    
    # Step 4: Evaluation
    logger.info("\n[STEP 4/4] Evaluating model...")
    metrics, y_test_actual, y_pred_actual = trainer.evaluate(
        model, X_test, y_test, scaler
    )
    
    # Save model metadata
    training_date = datetime.now().isoformat()
    trainer.save_metadata(metrics, training_date=training_date)
    
    # Save results visualization
    results_plot_path = os.path.join(
        settings.MODELS_DIR, 
        f"{model_name}_results.png"
    )
    plot_results(history, y_test_actual, y_pred_actual, metrics, results_plot_path)
    
    # Save metrics to file
    metrics_file = os.path.join(settings.MODELS_DIR, f"{model_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("MODEL TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ticker: {settings.TICKER}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Lookback Window: {settings.LOOKBACK_WINDOW} days\n")
        f.write(f"Train/Test Split: {settings.TRAIN_TEST_SPLIT:.0%}/{1-settings.TRAIN_TEST_SPLIT:.0%}\n")
        f.write(f"Epochs: {len(history.history['loss'])}\n")
        f.write(f"Batch Size: {settings.BATCH_SIZE}\n")
        f.write(f"Architecture: Stacked LSTM (4 layers, 96 units)\n")
        f.write(f"Approach: Univariate (Close price only)\n")
        f.write("\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"RMSE: ${metrics['rmse']:.2f}\n")
        f.write(f"MAE:  ${metrics['mae']:.2f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
        f.write(f"MSE:  ${metrics['mse']:.2f}\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"✓ Metrics saved to {metrics_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Model saved: {trainer.best_model_path}")
    logger.info(f"Results plot: {results_plot_path}")
    logger.info(f"Metrics file: {metrics_file}")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
