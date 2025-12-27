import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from config import settings
from src.dashboard_utils import (
    load_trained_model, load_stock_data, add_technical_indicators,
    make_predictions, predict_future, calculate_metrics,
    get_available_models, get_model_info
)

# Page configuration
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📈 Stock Forecast Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Stock Price Prediction using LSTM Neural Networks**")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock selection
    st.subheader("Stock Selection")
    ticker = st.text_input("Enter Stock Ticker", value=settings.TICKER, help="e.g., AAPL, GOOGL, MSFT")
    
    # Model selection
    st.subheader("Model Selection")
    available_models = get_available_models()
    
    if available_models:
        model_options = {f"{get_model_info(os.path.join(settings.MODELS_DIR, m))['name']} ({get_model_info(os.path.join(settings.MODELS_DIR, m))['date']})": m 
                        for m in available_models}
        selected_model_display = st.selectbox("Choose Model", list(model_options.keys()))
        selected_model = model_options[selected_model_display]
        model_path = os.path.join(settings.MODELS_DIR, selected_model)
    else:
        st.warning("⚠️ No trained models found!")
        st.info("Run `python main.py` to train a model first.")
        model_path = None
    
    # Prediction settings
    st.subheader("Prediction Settings")
    # Use fixed lookback from training configuration (cannot be changed)
    lookback = settings.LOOKBACK_WINDOW
    st.info(f"📊 **Lookback Window:** {lookback} days (fixed from training)")
    st.caption("The lookback window must match the training configuration and cannot be adjusted.")
    
    future_days = st.slider("Future Prediction Days", min_value=7, max_value=90, value=30, step=7,
                           help="Number of days to predict into the future")
    
    # Date range filter
    st.subheader("Visualization Period")
    days_to_show = st.selectbox("Show Data For", 
                                 ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
                                 index=3)
    
    st.divider()
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Info
    st.markdown("---")
    st.caption("📡 **Data Source:** Live from Yahoo Finance")
    st.caption(f"⏰ **Cache:** 5 minutes | **Updated:** {datetime.now().strftime('%H:%M:%S')}")

# Main content
if model_path and os.path.exists(model_path):
    # Load model and data
    with st.spinner("Loading model and data..."):
        model = load_trained_model(model_path)
        df = load_stock_data(ticker)
    
    if df is not None and len(df) > lookback:
        # Filter data based on selected period
        if days_to_show != "All Time":
            days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
            cutoff_date = datetime.now() - timedelta(days=days_map[days_to_show])
            df_display = df[df.index >= cutoff_date]
        else:
            df_display = df
        
        # Overview metrics
        st.header("📊 Overview")
        
        # Add live data indicator
        last_data_date = df.index[-1]
        data_age = (datetime.now() - last_data_date).days
        if data_age == 0:
            st.success("🟢 **Live Data** - Prices updated from Yahoo Finance")
        elif data_age == 1:
            st.info(f"🟡 **Recent Data** - Last updated: {last_data_date.strftime('%Y-%m-%d')} (1 day ago)")
        else:
            st.warning(f"🟠 **Delayed Data** - Last updated: {last_data_date.strftime('%Y-%m-%d')} ({data_age} days ago)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", 
                     f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        with col2:
            st.metric("52-Week High", f"${df['High'].tail(252).max():.2f}")
        
        with col3:
            st.metric("52-Week Low", f"${df['Low'].tail(252).min():.2f}")
        
        with col4:
            avg_volume = df['Volume'].tail(30).mean()
            st.metric("Avg Volume (30d)", f"{avg_volume/1e6:.2f}M")
        
        st.divider()
        
        # Historical price chart with technical indicators
        st.header("📈 Historical Price & Technical Indicators")
        
        df_with_indicators = add_technical_indicators(df_display)
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Moving Averages', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_with_indicators.index,
                open=df_with_indicators['Open'],
                high=df_with_indicators['High'],
                low=df_with_indicators['Low'],
                close=df_with_indicators['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['SMA_20'],
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=1, col=1
            )
        
        if 'SMA_50' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['SMA_50'],
                          name='SMA 50', line=dict(color='blue', width=1)),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BBL_20_2.0' in df_with_indicators.columns and 'BBU_20_2.0' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BBU_20_2.0'],
                          name='BB Upper', line=dict(color='gray', width=1, dash='dash'),
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['BBL_20_2.0'],
                          name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                          fill='tonexty', fillcolor='rgba(128, 128, 128, 0.1)'),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if df_with_indicators['Close'].iloc[i] < df_with_indicators['Open'].iloc[i] 
                  else 'green' for i in range(len(df_with_indicators))]
        fig.add_trace(
            go.Bar(x=df_with_indicators.index, y=df_with_indicators['Volume'],
                   name='Volume', marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        # RSI
        if 'RSI_14' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['RSI_14'],
                          name='RSI', line=dict(color='purple', width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'MACD_12_26_9' in df_with_indicators.columns:
            fig.add_trace(
                go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACD_12_26_9'],
                          name='MACD', line=dict(color='blue', width=2)),
                row=4, col=1
            )
            if 'MACDs_12_26_9' in df_with_indicators.columns:
                fig.add_trace(
                    go.Scatter(x=df_with_indicators.index, y=df_with_indicators['MACDs_12_26_9'],
                              name='Signal', line=dict(color='orange', width=2)),
                    row=4, col=1
                )
            if 'MACDh_12_26_9' in df_with_indicators.columns:
                fig.add_trace(
                    go.Bar(x=df_with_indicators.index, y=df_with_indicators['MACDh_12_26_9'],
                          name='Histogram', marker_color='gray'),
                    row=4, col=1
                )
        
        fig.update_layout(
            height=1000,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Predictions section
        st.header("🔮 Model Predictions")
        
        with st.spinner("Generating predictions..."):
            predictions, actual_values, prediction_dates = make_predictions(model, df, lookback)
        
        if predictions is not None:
            # Filter predictions for display period
            if days_to_show != "All Time":
                days_map = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
                cutoff_date = datetime.now() - timedelta(days=days_map[days_to_show])
                mask = prediction_dates >= cutoff_date
                prediction_dates_display = prediction_dates[mask]
                predictions_display = predictions[mask]
                actual_values_display = actual_values[mask]
            else:
                prediction_dates_display = prediction_dates
                predictions_display = predictions
                actual_values_display = actual_values
            
            # Calculate metrics
            metrics = calculate_metrics(actual_values_display, predictions_display)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RMSE", f"${metrics['RMSE']:.2f}")
            with col2:
                st.metric("MAE", f"${metrics['MAE']:.2f}")
            with col3:
                st.metric("MSE", f"${metrics['MSE']:.2f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            
            # Actual vs Predicted chart
            fig_pred = go.Figure()
            
            fig_pred.add_trace(
                go.Scatter(x=prediction_dates_display, y=actual_values_display,
                          name='Actual Price', line=dict(color='blue', width=2))
            )
            
            fig_pred.add_trace(
                go.Scatter(x=prediction_dates_display, y=predictions_display,
                          name='Predicted Price', line=dict(color='red', width=2, dash='dash'))
            )
            
            fig_pred.update_layout(
                title="Actual vs Predicted Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Prediction error distribution
            errors = actual_values_display - predictions_display
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_error = go.Figure()
                fig_error.add_trace(
                    go.Histogram(x=errors, nbinsx=30, name='Prediction Errors',
                                marker_color='indianred')
                )
                fig_error.update_layout(
                    title="Prediction Error Distribution",
                    xaxis_title="Error ($)",
                    yaxis_title="Frequency",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_error, use_container_width=True)
            
            with col2:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(
                    go.Scatter(x=actual_values_display, y=predictions_display,
                              mode='markers', name='Predictions',
                              marker=dict(size=5, color='blue', opacity=0.6))
                )
                # Add diagonal line
                min_val = min(actual_values_display.min(), predictions_display.min())
                max_val = max(actual_values_display.max(), predictions_display.max())
                fig_scatter.add_trace(
                    go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                              mode='lines', name='Perfect Prediction',
                              line=dict(color='red', dash='dash'))
                )
                fig_scatter.update_layout(
                    title="Actual vs Predicted (Scatter)",
                    xaxis_title="Actual Price ($)",
                    yaxis_title="Predicted Price ($)",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.divider()
        
        # Future predictions
        st.header("🚀 Future Price Forecast")
        
        with st.spinner(f"Predicting next {future_days} days..."):
            future_prices, future_dates = predict_future(model, df, days_ahead=future_days, lookback=lookback)
        
        if future_prices is not None:
            # Display future prediction
            last_actual_price = df['Close'].iloc[-1]
            first_predicted_price = future_prices[0]
            last_predicted_price = future_prices[-1]
            predicted_change = last_predicted_price - last_actual_price
            predicted_change_pct = (predicted_change / last_actual_price) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${last_actual_price:.2f}")
            with col2:
                st.metric(f"Predicted Price ({future_days}d)", f"${last_predicted_price:.2f}",
                         f"{predicted_change:+.2f} ({predicted_change_pct:+.2f}%)")
            with col3:
                direction = "📈 Bullish" if predicted_change > 0 else "📉 Bearish"
                st.metric("Market Sentiment", direction)
            
            # Future prediction chart
            fig_future = go.Figure()
            
            # Historical prices (last 60 days)
            historical_cutoff = df.index[-60:] if len(df) >= 60 else df.index
            fig_future.add_trace(
                go.Scatter(x=historical_cutoff, y=df.loc[historical_cutoff, 'Close'],
                          name='Historical Price', line=dict(color='blue', width=2))
            )
            
            # Future predictions
            fig_future.add_trace(
                go.Scatter(x=future_dates, y=future_prices,
                          name='Predicted Price', line=dict(color='red', width=2, dash='dash'),
                          mode='lines+markers')
            )
            
            # Add shaded confidence region (simplified as ±5% for visualization)
            upper_bound = future_prices * 1.05
            lower_bound = future_prices * 0.95
            
            fig_future.add_trace(
                go.Scatter(x=future_dates, y=upper_bound,
                          name='Upper Bound', line=dict(width=0),
                          showlegend=False, hoverinfo='skip')
            )
            fig_future.add_trace(
                go.Scatter(x=future_dates, y=lower_bound,
                          name='Confidence Interval', fill='tonexty',
                          fillcolor='rgba(255, 0, 0, 0.2)',
                          line=dict(width=0))
            )
            
            fig_future.update_layout(
                title=f"{ticker} Price Forecast - Next {future_days} Days",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Future predictions table
            with st.expander("📋 View Detailed Forecast Table"):
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_prices,
                    'Change from Today': future_prices - last_actual_price,
                    'Change %': ((future_prices - last_actual_price) / last_actual_price) * 100
                })
                future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
                future_df['Predicted Price'] = future_df['Predicted Price'].apply(lambda x: f"${x:.2f}")
                future_df['Change from Today'] = future_df['Change from Today'].apply(lambda x: f"${x:+.2f}")
                future_df['Change %'] = future_df['Change %'].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(future_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Additional info
        with st.expander("ℹ️ About This Dashboard"):
            st.markdown(f"""
            ### Stock Forecast Dashboard
            
            This dashboard uses a **Long Short-Term Memory (LSTM)** neural network to predict stock prices.
            
            **Features:**
            - **Live stock data** fetched from Yahoo Finance API
            - Real-time price updates (cached for 5 minutes)
            - Technical indicators visualization (RSI, MACD, Bollinger Bands, Moving Averages)
            - Historical prediction accuracy metrics
            - Future price forecasting
            
            **Model Details:**
            - Architecture: Stacked LSTM (4 layers, 96 units each) with dropout
            - Approach: **Univariate** (Close price only, no technical indicators)
            - Lookback window: **{lookback} days** (fixed from training)
            - Based on research findings showing univariate models outperform multivariate
            
            **Data Source:**
            - Live prices from Yahoo Finance
            - Historical data merged with CSV cache
            - Auto-refresh every 5 minutes
            
            **Disclaimer:** This is for educational purposes only. Do not use for actual investment decisions.
            Stock predictions are inherently uncertain and past performance does not guarantee future results.
            """)
    
    elif df is None:
        st.error(f"❌ No data found for ticker '{ticker}'. Please run the data ingestion pipeline first.")
        st.info("Run: `python main.py` to fetch and process stock data.")
    else:
        st.error(f"❌ Insufficient data for predictions. Need at least {lookback} days of historical data.")

else:
    st.warning("⚠️ No trained model available!")
    st.info("""
    **To get started:**
    1. Run `python main.py` to train a stock forecast model
    2. Refresh this dashboard to load the trained model
    """)
