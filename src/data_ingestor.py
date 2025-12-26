# src/data_ingestor.py
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

class DataIngestor:
    def __init__(self, ticker, config):
        self.ticker = ticker
        self.data_dir = config['data_dirs']['raw']
        self.file_path = f"{self.data_dir}/{ticker}_raw.csv"

    def fetch_and_validate(self):
        """Fetches new data since last update, appends, and validates."""
        
        # 1. Determine the start date for new data (last date in file + 1)
        if os.path.exists(self.file_path):
            existing_data = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)
            last_date = existing_data.index.max()
            start_date = last_date + timedelta(days=1)
        else:
            existing_data = None
            start_date = datetime(2020, 1, 1)  # Initial start date from config

        # 2. Only fetch new data if needed
        if start_date.date() < datetime.now().date():
            self.logger.info(f"Fetching new data from {start_date.date()}")
            new_data = yf.download(self.ticker, start=start_date, end=datetime.now())
            
            if not new_data.empty:
                # 3. Append to existing data
                if existing_data is not None:
                    updated_data = pd.concat([existing_data, new_data])
                else:
                    updated_data = new_data
                
                # 4. Basic validation (no missing dates, negative volumes)
                updated_data = self._validate_data(updated_data)
                
                # 5. Save updated dataset
                updated_data.to_csv(self.file_path)
                self.logger.info(f"Data saved to {self.file_path}")
            else:
                self.logger.info("No new data available.")
        
        return self.file_path

    def _validate_data(self, df):
        """Performs data quality checks."""
        # Check for missing dates
        all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        missing_dates = all_dates.difference(df.index)
        if len(missing_dates) > 0:
            self.logger.warning(f"{len(missing_dates)} trading dates missing.")
        
        # Check for absurd values
        if (df['Volume'] < 0).any():
            raise ValueError("Negative volume found.")
        
        return df