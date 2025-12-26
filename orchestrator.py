# orchestrator.py
import logging
from datetime import datetime
import yaml

# Import your custom modules
from src.data_ingestor import DataIngestor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_registry import ModelRegistry

class PipelineOrchestrator:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.ticker = self.config['ticker']
        self.logger = self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            filename=f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def run(self):
        self.logger.info(f"Starting pipeline for {self.ticker}")
        
        try:
            # 1. INGEST: Fetch and validate new data
            self.logger.info("Step 1: Data Ingestion")
            ingestor = DataIngestor(self.ticker, self.config)
            raw_data_path = ingestor.fetch_and_validate()
            
            # 2. PROCESS: Engineer features and create sequences
            self.logger.info("Step 2: Feature Engineering")
            engineer = FeatureEngineer(raw_data_path, self.config)
            processed_data_path, sequences = engineer.process()
            
            # 3. TRAIN: Train a new model and evaluate it
            self.logger.info("Step 3: Model Training")
            trainer = ModelTrainer(sequences, self.config)
            new_model_path, metrics = trainer.train_and_evaluate()
            
            # 4. DEPLOY: Validate and promote if better than current
            self.logger.info("Step 4: Model Validation & Promotion")
            registry = ModelRegistry(self.config)
            if registry.validate_and_promote(new_model_path, metrics):
                self.logger.info("New model promoted to production.")
            else:
                self.logger.info("New model rejected. Keeping current production model.")
                
            self.logger.info("Pipeline finished successfully.")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise

if __name__ == "__main__":
    pipeline = PipelineOrchestrator()
    pipeline.run()