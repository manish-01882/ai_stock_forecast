# src/model_trainer_paper.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import json
from datetime import datetime
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainerPaper:
    """
    LSTM Model Trainer based on Research Paper Architecture.
    
    Architecture from Paper 1:
    - 4 Stacked LSTM layers (96 units each)
    - Dropout (0.2) after each LSTM layer
    - 1 Dense output layer
    - Adam optimizer with MSE loss
    - 50-100 epochs training
    """
    def __init__(self, model_name=None, ticker=None):
        self.model_name = model_name or f"lstm_paper_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.ticker = ticker  # Store ticker for metadata
        self.model_path = os.path.join(settings.MODELS_DIR, f"{self.model_name}.h5")
        self.best_model_path = os.path.join(settings.MODELS_DIR, f"{self.model_name}_best.h5")
        self.metadata_path = os.path.join(settings.MODELS_DIR, f"{self.model_name}.json")

    def build_model(self, input_shape):
        """Builds a Stacked LSTM model exactly as described in Paper 1.
        
        Architecture:
        - Layer 1: LSTM (96 units, return_sequences=True)
        - Dropout (0.2)
        - Layer 2: LSTM (96 units, return_sequences=True)
        - Dropout (0.2)
        - Layer 3: LSTM (96 units, return_sequences=True)
        - Dropout (0.2)
        - Layer 4: LSTM (96 units, return_sequences=False)
        - Dropout (0.2)
        - Dense (1 unit) - Output layer
        
        Args:
            input_shape: Tuple (lookback, features) e.g., (50, 1) for univariate
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First LSTM layer
            LSTM(units=96, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=96, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(units=96, return_sequences=True),
            Dropout(0.2),
            
            # Fourth LSTM layer
            LSTM(units=96, return_sequences=False),
            Dropout(0.2),
            
            # Output layer
            Dense(units=1)
        ])
        
        # Compile with Adam optimizer and MSE loss (as per papers)
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        logger.info("✓ Stacked LSTM model built (Paper 1 architecture)")
        logger.info(f"  Total parameters: {model.count_params():,}")
        logger.info(f"  Architecture: 4 LSTM layers (96 units each) + Dropout (0.2)")
        
        return model

    def get_callbacks(self):
        """Returns training callbacks.
        
        Based on settings.USE_EARLY_STOPPING:
        - If True: Uses early stopping (current approach)
        - If False: Only uses ModelCheckpoint (paper approach - train full epochs)
        """
        callbacks = [
            # Always save the best model
            ModelCheckpoint(
                filepath=self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            )
        ]
        
        if settings.USE_EARLY_STOPPING:
            logger.info("  Early stopping: ENABLED")
            callbacks.extend([
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min'
                )
            ])
        else:
            logger.info("  Early stopping: DISABLED (training full epochs as per papers)")
        
        return callbacks

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=None, batch_size=None):
        """Trains the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs (default from settings)
            batch_size: Batch size (default from settings)
            
        Returns:
            model, history
        """
        epochs = epochs or settings.EPOCHS
        batch_size = batch_size or settings.BATCH_SIZE
        
        logger.info(f"Training model on data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape if X_val is not None else 'None'}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")

        model = self.build_model((X_train.shape[1], X_train.shape[2]))

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train the model
        logger.info(f"Starting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save the final model
        model.save(self.model_path)
        logger.info(f"✓ Final model saved to {self.model_path}")
        logger.info(f"✓ Best model saved to {self.best_model_path}")
        
        # Load the best model for return
        from tensorflow.keras.models import load_model
        if os.path.exists(self.best_model_path):
            model = load_model(self.best_model_path)
            logger.info("✓ Loaded best model from training")
        
        return model, history

    def evaluate(self, model, X_test, y_test, scaler):
        """Evaluates model performance and returns metrics.
        
        Args:
            model: Trained Keras model
            X_test: Test sequences
            y_test: Test targets (normalized)
            scaler: MinMaxScaler used for normalization
            
        Returns:
            metrics dict, actual prices, predicted prices
        """
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform the scaling to get actual prices
        # For univariate: scaler expects shape (n, 1)
        y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Calculate error metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

        logger.info("=" * 60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"  RMSE: ${rmse:.2f}")
        logger.info(f"  MAE:  ${mae:.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info("=" * 60)
        
        return {
            "rmse": rmse, 
            "mae": mae, 
            "mse": mse, 
            "mape": mape
        }, y_test_actual, y_pred_actual
    
    def save_metadata(self, metrics, training_date=None):
        """Save model metadata as JSON file.
        
        Args:
            metrics: Dict containing rmse, mae, mse, mape
            training_date: Training timestamp (defaults to now)
        """
        metadata = {
            "ticker": self.ticker or "UNKNOWN",
            "training_date": training_date or datetime.now().isoformat(),
            "lookback_window": settings.LOOKBACK_WINDOW,
            "architecture": "Stacked LSTM (4 layers, 96 units)",
            "approach": "univariate",
            "dropout": 0.2,
            "epochs": settings.EPOCHS,
            "batch_size": settings.BATCH_SIZE,
            "metrics": {
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mae"]),
                "mse": float(metrics["mse"]),
                "mape": float(metrics["mape"])
            },
            "model_path": self.best_model_path,
            "model_name": self.model_name
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ Metadata saved to {self.metadata_path}")
        return metadata


if __name__ == "__main__":
    # Quick test of model architecture
    from config import settings
    
    logger.info("Testing model architecture...")
    trainer = ModelTrainerPaper()
    
    # Build model with univariate input shape (50 timesteps, 1 feature)
    model = trainer.build_model((settings.LOOKBACK_WINDOW, 1))
    model.summary()
