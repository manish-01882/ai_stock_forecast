# main.py
import logging
from datetime import datetime
from src.data_ingestor import DataIngestor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{settings.LOGS_DIR}/pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline():
    """Orchestrates the entire ML pipeline."""
    logger.info("="*50)
    logger.info("Starting Stock Forecast Pipeline")
    logger.info("="*50)

    try:
        # Step 1: Ingest Data
        logger.info("STEP 1: Data Ingestion")
        ingestor = DataIngestor()
        raw_data_path = ingestor.fetch_and_save()

        # Step 2: Feature Engineering
        logger.info("STEP 2: Feature Engineering")
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test, scaler = engineer.process(raw_data_path)

        # Step 3: Model Training
        logger.info("STEP 3: Model Training")
        trainer = ModelTrainer()
        model, history = trainer.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=64)

        # Step 4: Model Evaluation
        logger.info("STEP 4: Model Evaluation")
        metrics, y_actual, y_pred = trainer.evaluate(model, X_test, y_test, scaler)

        logger.info(f"Pipeline completed successfully. Model saved as: {trainer.model_path}")
        logger.info(f"Final Test RMSE: ${metrics['rmse']:.2f}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline()