# src/data_ingestor.py
import sys
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestor:
    def __init__(self, ticker=settings.TICKER):
        self.ticker = ticker
        self.file_path = os.path.join(settings.DATA_RAW_DIR, f"{ticker}_raw.csv")

    def fetch_and_save(self):
        """Fetches new data and appends to existing CSV."""
        try:
            # Ensure data directory exists
            os.makedirs(settings.DATA_RAW_DIR, exist_ok=True)
            
            # Determine the start date for new data
            if os.path.exists(self.file_path):
                existing_df = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)
                # Keep only main columns, drop any duplicates
                main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
                existing_df = existing_df[[col for col in main_cols if col in existing_df.columns]]
                last_date = existing_df.index.max()
                start_date = last_date + pd.Timedelta(days=1)
                logger.info(f"Found existing data until {last_date.date()}")
            else:
                existing_df = None
                start_date = settings.START_DATE
                logger.info("No existing data found. Starting fresh.")

            # If start_date is in the past, fetch new data
            if pd.to_datetime(start_date) < datetime.now():
                logger.info(f"Fetching data from {start_date} for ticker {self.ticker}")
                new_data = yf.download(self.ticker, start=start_date, progress=False)

                if new_data is None or new_data.empty:
                    logger.warning(f"No data returned from yfinance for {self.ticker}. Check ticker validity.")
                    if existing_df is not None and not existing_df.empty:
                        logger.info("Using existing data instead.")
                        return self.file_path
                    else:
                        raise ValueError(f"Failed to fetch data for ticker {self.ticker}. Please verify the ticker symbol is valid.")

                # Flatten multi-index columns if present (yfinance sometimes returns multi-index)
                if isinstance(new_data.columns, pd.MultiIndex):
                    new_data.columns = new_data.columns.get_level_values(0)
                
                # Keep only main columns
                main_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
                new_data = new_data[[col for col in main_cols if col in new_data.columns]]
                
                # Append new data to existing or create new file
                if existing_df is not None:
                    updated_df = pd.concat([existing_df, new_data])
                    updated_df = updated_df[~updated_df.index.duplicated(keep='last')]  # Remove duplicates
                else:
                    updated_df = new_data

                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
                updated_df.to_csv(self.file_path)
                logger.info(f"✓ Data saved to {self.file_path}. Total rows: {len(updated_df)}, New rows: {len(new_data)}")
            else:
                logger.info("Data is already up to date.")

            return self.file_path

        except Exception as e:
            logger.error(f"Error in data ingestion for {self.ticker}: {e}")
            raise

if __name__ == "__main__":
    ingestor = DataIngestor()
    ingestor.fetch_and_save()