#!/usr/bin/env python3
"""
train_batch.py

Batch training script for multiple stock tickers.
Trains LSTM models for all tickers defined in config.yaml.
"""
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from datetime import datetime
import logging
from tabulate import tabulate

from src.data_ingestor import DataIngestor
from src.feature_engineer import FeatureEngineerUnivariate
from src.model_trainer import ModelTrainerPaper
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model_for_ticker(ticker):
    """Train a model for a specific ticker.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
        
    Returns:
        dict: Training results with model_path and metrics, or None if failed
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING MODEL FOR {ticker}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Data Ingestion
        logger.info(f"\n[1/4] Fetching stock data for {ticker}...")
        ingestor = DataIngestor(ticker=ticker)
        raw_file_path = ingestor.fetch_and_save()
        
        # Step 2: Feature Engineering (Univariate)
        logger.info(f"\n[2/4] Processing data (univariate approach)...")
        feature_engineer = FeatureEngineerUnivariate(
            lookback=settings.LOOKBACK_WINDOW,
            price_column='Close'
        )
        X_train, X_test, y_train, y_test, scaler = feature_engineer.process(raw_file_path)
        
        # Create validation set from training data (80/20 split)
        val_split = int(len(X_train) * 0.8)
        X_train_final = X_train[:val_split]
        y_train_final = y_train[:val_split]
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        
        logger.info(f"  Final training set: {X_train_final.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        logger.info(f"  Test set: {X_test.shape}")
        
        # Step 3: Model Training
        logger.info(f"\n[3/4] Training LSTM model...")
        model_name = f"lstm_paper_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        trainer = ModelTrainerPaper(model_name=model_name, ticker=ticker)
        
        model, history = trainer.train(
            X_train_final, y_train_final,
            X_val, y_val,
            epochs=settings.EPOCHS,
            batch_size=settings.BATCH_SIZE
        )
        
        # Step 4: Evaluation
        logger.info(f"\n[4/4] Evaluating model...")
        metrics, y_test_actual, y_pred_actual = trainer.evaluate(
            model, X_test, y_test, scaler
        )
        
        # Save metadata
        training_date = datetime.now().isoformat()
        trainer.save_metadata(metrics, training_date=training_date)
        
        logger.info(f"✓ {ticker} model training completed successfully!")
        
        return {
            'ticker': ticker,
            'model_path': trainer.best_model_path,
            'metrics': metrics,
            'status': 'SUCCESS'
        }
        
    except Exception as e:
        logger.error(f"✗ Failed to train model for {ticker}: {e}", exc_info=True)
        return {
            'ticker': ticker,
            'model_path': None,
            'metrics': None,
            'status': 'FAILED',
            'error': str(e)
        }


def main():
    """Main batch training pipeline."""
    logger.info("=" * 80)
    logger.info("BATCH TRAINING - MULTI-STOCK LSTM MODELS")
    logger.info("=" * 80)
    logger.info(f"Tickers to train: {', '.join(settings.TICKERS)}")
    logger.info(f"Lookback Window: {settings.LOOKBACK_WINDOW} days")
    logger.info(f"Epochs: {settings.EPOCHS}")
    logger.info(f"Batch Size: {settings.BATCH_SIZE}")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    results = []
    
    # Train models for each ticker
    for idx, ticker in enumerate(settings.TICKERS, 1):
        logger.info(f"\n\n{'#' * 80}")
        logger.info(f"# Training {idx}/{len(settings.TICKERS)}: {ticker}")
        logger.info(f"{'#' * 80}\n")
        
        result = train_model_for_ticker(ticker)
        results.append(result)
    
    # Generate summary report
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("\n\n" + "=" * 80)
    logger.info("BATCH TRAINING SUMMARY")
    logger.info("=" * 80)
    
    # Prepare summary table
    table_data = []
    for result in results:
        if result['status'] == 'SUCCESS':
            metrics = result['metrics']
            table_data.append([
                result['ticker'],
                result['status'],
                f"${metrics['rmse']:.2f}",
                f"${metrics['mae']:.2f}",
                f"{metrics['mape']:.2f}%"
            ])
        else:
            table_data.append([
                result['ticker'],
                result['status'],
                'N/A',
                'N/A',
                'N/A'
            ])
    
    print("\n" + tabulate(
        table_data,
        headers=['Ticker', 'Status', 'RMSE', 'MAE', 'MAPE'],
        tablefmt='grid'
    ))
    
    # Summary statistics
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = len(results) - success_count
    
    logger.info(f"\nTotal models trained: {len(results)}")
    logger.info(f"✓ Successful: {success_count}")
    logger.info(f"✗ Failed: {failed_count}")
    logger.info(f"Total time: {duration}")
    logger.info("=" * 80)
    
    # Save summary to file
    summary_file = os.path.join(settings.MODELS_DIR, f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    with open(summary_file, 'w') as f:
        f.write("BATCH TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Tickers: {', '.join(settings.TICKERS)}\n")
        f.write("\n" + tabulate(
            table_data,
            headers=['Ticker', 'Status', 'RMSE', 'MAE', 'MAPE'],
            tablefmt='grid'
        ) + "\n")
        f.write(f"\nTotal: {len(results)} | Success: {success_count} | Failed: {failed_count}\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"Summary saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        # Exit with error code if any training failed
        failed_count = sum(1 for r in results if r['status'] == 'FAILED')
        if failed_count > 0:
            logger.warning(f"\n⚠️  {failed_count} model(s) failed to train. Check logs above.")
            sys.exit(1)
        else:
            logger.info("\n✓ All models trained successfully!")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Batch training failed: {e}", exc_info=True)
        sys.exit(1)
