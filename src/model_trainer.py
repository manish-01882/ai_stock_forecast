# src/model_trainer.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
from datetime import datetime
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name=None):
        self.model_name = model_name or f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.model_path = os.path.join(settings.MODELS_DIR, f"{self.model_name}.h5")
        self.best_model_path = os.path.join(settings.MODELS_DIR, f"{self.model_name}_best.h5")

    def build_model(self, input_shape):
        """Builds a balanced LSTM model - complex enough to learn, simple enough to generalize."""
        model = Sequential([
            # First Bidirectional LSTM layer (reduced from 128 to 96)
            Bidirectional(LSTM(units=96, return_sequences=True, 
                              kernel_regularizer=l2(0.001)), 
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.4),  # Increased dropout to prevent overfitting
            
            # Second Bidirectional LSTM layer (reduced from 64 to 48)
            Bidirectional(LSTM(units=48, return_sequences=False,
                              kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            Dropout(0.4),
            
            # Dense layers (simplified)
            Dense(units=24, activation='relu', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(units=1)  # Output layer for price prediction
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        logger.info("Balanced LSTM model built successfully.")
        logger.info(f"Total parameters: {model.count_params():,}")
        return model

    def get_callbacks(self):
        """Returns training callbacks for better model training."""
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # Save the best model during training
            ModelCheckpoint(
                filepath=self.best_model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            )
        ]
        return callbacks

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Trains the LSTM model with callbacks."""
        logger.info(f"Training model on data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape if X_val is not None else 'None'}")

        model = self.build_model((X_train.shape[1], X_train.shape[2]))

        # Get callbacks
        callbacks = self.get_callbacks()

        # Train the model
        logger.info(f"Starting training for up to {epochs} epochs with early stopping...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        # Save the final model (though best model is already saved by checkpoint)
        model.save(self.model_path)
        logger.info(f"Final model saved to {self.model_path}")
        logger.info(f"Best model saved to {self.best_model_path}")
        
        # Load the best model for return
        from tensorflow.keras.models import load_model
        if os.path.exists(self.best_model_path):
            model = load_model(self.best_model_path)
            logger.info("Loaded best model from training")
        
        return model, history

    def evaluate(self, model, X_test, y_test, scaler):
        """Evaluates model performance and returns metrics."""
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform the scaling to get actual prices
        # Create dummy arrays for inverse transform (we only need the 'Close' price column)
        dummy_array_pred = np.zeros((len(y_pred), scaler.n_features_in_))
        dummy_array_pred[:, 0] = y_pred.flatten()
        y_pred_actual = scaler.inverse_transform(dummy_array_pred)[:, 0]

        dummy_array_test = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_array_test[:, 0] = y_test.flatten()
        y_test_actual = scaler.inverse_transform(dummy_array_test)[:, 0]

        # Calculate error metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

        logger.info(f"Test RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")
        return {"rmse": rmse, "mae": mae, "mse": mse, "mape": mape}, y_test_actual, y_pred_actual