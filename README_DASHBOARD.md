# Stock Forecast Dashboard - Quick Start Guide

## 📋 Overview

This Streamlit dashboard provides an interactive interface for visualizing and analyzing stock price predictions using the trained LSTM model.

## 🚀 Getting Started

### Prerequisites

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models
```bash
# Train all tickers at once (AAPL, GOOGL, MSFT)
python train_batch.py

# Or train a single ticker
python train_model.py
```

### 2. Run Dashboard
```bash
streamlit run app.py
```

### 3. Access
Open browser to `http://localhost:8501`

---

## Project Structure

```
stock_forecast/
├── app.py                    # Streamlit dashboard
├── train_batch.py           # Batch training for multiple tickers
├── train_model.py           # Single ticker training
├── requirements.txt
├── config/
│   ├── config.yaml         # Multi-ticker configuration
│   └── settings.py         # Global settings
├── src/
│   ├── data_ingestor.py
│   ├── feature_engineer.py
│   ├── model_trainer.py
│   └── dashboard_utils.py
├── data/
│   └── raw/                # Stock CSV files
├── models/                 # Trained models & metadata
└── logs/
```

## 🎯 Dashboard Features

### 1. **Configuration Panel (Sidebar)**
   - **Stock Ticker Selection**: Choose which stock to analyze
   - **Model Selection**: Pick from available trained models
   - **Lookback Window**: Adjust the historical window (30-150 days)
   - **Future Prediction Days**: Set forecast horizon (7-90 days)
   - **Visualization Period**: Filter data display range

### 2. **Overview Section**
   - Current stock price with daily change
   - 52-week high and low
   - Average trading volume

### 3. **Historical Price & Technical Indicators**
   - **Candlestick Chart**: OHLC (Open, High, Low, Close) visualization
   - **Moving Averages**: SMA 20 and SMA 50
   - **Bollinger Bands**: Volatility indicator
   - **Volume Chart**: Trading volume with color-coded bars
   - **RSI (Relative Strength Index)**: Momentum indicator
   - **MACD**: Trend-following momentum indicator

### 4. **Model Predictions**
   - **Actual vs Predicted Comparison**: Line chart comparing model predictions with actual prices
   - **Performance Metrics**: RMSE, MAE, MSE, MAPE
   - **Error Distribution**: Histogram showing prediction errors
   - **Scatter Plot**: Actual vs predicted correlation

### 5. **Future Price Forecast**
   - **Price Prediction**: Forecast for next N days (configurable)
   - **Confidence Interval**: Shaded region showing uncertainty
   - **Market Sentiment**: Bullish/bearish indicator
   - **Detailed Forecast Table**: Expandable table with daily predictions

## 📊 Dashboard Usage Tips

### Analyzing Stock Performance
1. Select your desired stock ticker in the sidebar
2. Choose a visualization period (e.g., "1 Year")
3. Review the technical indicators for trend analysis
4. Check RSI for overbought (>70) or oversold (<30) conditions
5. Use MACD crossovers to identify potential buy/sell signals

### Evaluating Model Accuracy
1. Check the **RMSE** and **MAE** metrics - lower is better
2. Review the **Actual vs Predicted** chart for visual accuracy
3. Examine the **Error Distribution** - should be centered around 0
4. Look at the **Scatter Plot** - points should follow the diagonal line

### Making Predictions
1. Adjust the **Future Prediction Days** slider
2. View the **Future Price Forecast** chart
3. Check the predicted price change and percentage
4. Expand the **Detailed Forecast Table** for specific dates
5. Consider the confidence interval (shaded area) for uncertainty

## ⚙️ Configuration Options

### Changing the Default Stock
Edit `config/settings.py`:
```python
TICKER = "GOOGL"  # Change to your preferred stock
```

### Adjusting Model Parameters
Edit `config/settings.py`:
```python
LOOKBACK_WINDOW = 100  # Historical window
TRAIN_TEST_SPLIT = 0.65  # Train/test ratio
```

## 🔄 Refreshing Data

To update with the latest stock data:
1. Click the **🔄 Refresh Data** button in the sidebar
2. Or re-run the training pipeline: `python main.py`

## 📝 Important Notes

### Data Requirements
- The dashboard requires at least **lookback window** days of historical data
- Run `python main.py` at least once before launching the dashboard
- Data is cached for performance - use refresh button to clear cache

### Model Files
- Models are stored in the `models/` directory
- The dashboard auto-detects all `.h5` model files
- Most recent models appear first in the dropdown

### Performance Optimization
- The dashboard uses Streamlit caching (`@st.cache_data`, `@st.cache_resource`)
- First load may be slow, subsequent interactions are fast
- Use the refresh button if you've added new data or models

## 🎨 Customization

### Changing Visual Theme
Streamlit supports theme customization via `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Adding More Technical Indicators
Edit `src/dashboard_utils.py` in the `add_technical_indicators()` function:
```python
df_copy.ta.ema(length=12, append=True)  # Exponential Moving Average
df_copy.ta.atr(length=14, append=True)  # Average True Range
```

## 🐛 Troubleshooting

### Issue: "No trained models found"
**Solution**: Run `python main.py` to train a model first

### Issue: "No data found for ticker"
**Solution**: 
1. Check if the ticker symbol is correct
2. Run `python main.py` or check `data/raw/` directory
3. Ensure internet connection for Yahoo Finance API

### Issue: "Insufficient data for predictions"
**Solution**: Reduce the lookback window or collect more historical data

### Issue: Dashboard is slow
**Solution**: 
1. Reduce the visualization period (e.g., from "All Time" to "1 Year")
2. Lower the future prediction days
3. Click refresh to clear cache if memory is an issue

## 📚 Further Reading

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Graphing Library](https://plotly.com/python/)
- [LSTM for Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Technical Analysis Indicators](https://github.com/twopirllc/pandas-ta)

## ⚠️ Disclaimer

This dashboard is for **educational and research purposes only**. 

- Stock price predictions are inherently uncertain
- Past performance does not guarantee future results
- Do NOT use this for actual investment decisions
- Consult a financial advisor for investment advice

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main project `README.md`
3. Examine logs in the `logs/` directory

---

**Happy Analyzing! 📈**
