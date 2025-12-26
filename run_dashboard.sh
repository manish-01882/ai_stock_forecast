#!/bin/bash

# Stock Forecast Dashboard Launcher
# This script launches the Streamlit dashboard

echo "🚀 Launching Stock Forecast Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Check if data exists
if [ ! -f "data/raw/AAPL_raw.csv" ]; then
    echo "⚠️  Warning: No stock data found!"
    echo "📊 Running data pipeline to fetch stock data..."
    python main.py
    echo ""
fi

# Launch the dashboard
echo "✅ Starting dashboard at http://localhost:8501"
echo "📈 Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
