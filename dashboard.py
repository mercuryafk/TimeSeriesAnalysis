# -*- coding: utf-8 -*-
"""
Time Series Stock Market Forecasting Dashboard (Pre-computed)

This dashboard loads pre-computed forecasts from CSV files and visualizes them.
It is designed to be fast and lightweight for deployment.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    layout="wide"
)


# --- Data Loading Functions ---
@st.cache_data
def load_data(file_path):
    """Loads and robustly cleans the standard data/forecast CSV files."""
    try:
        df = pd.read_csv(file_path)
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        df.set_index('Date', inplace=True)
        df.index = df.index.normalize()
        return df
    except FileNotFoundError:
        return None


@st.cache_data
def load_prophet_data(file_path):
    """Loads the pre-computed Prophet forecast CSV which has a 'ds' column."""
    try:
        df = pd.read_csv(file_path, parse_dates=['ds'])
        return df
    except FileNotFoundError:
        return None


# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg", width=60)
    st.markdown("---")
    st.header("About this Project")
    st.info(
        "This dashboard presents a comparative analysis of four different time series "
        "forecasting models for Apple (AAPL) stock prices. All forecasts are pre-computed "
        "and loaded for instant visualization. "
    )
    st.subheader("Models Implemented:")
    st.markdown(
        """
        - **ARIMA** (Autoregressive Integrated Moving Average)
        - **SARIMA** (Seasonal ARIMA)
        - **Prophet** (by Meta)
        - **LSTM** (Long Short-Term Memory Network)
        """
    )
    st.markdown("---")
    st.subheader("Data Source")
    st.markdown("Historical stock data is sourced from Yahoo Finance via the `yfinance` library. Data has been taken for 10 years from 2015 to 2025.")
    st.markdown("---")
    # IMPORTANT: Remember to replace this with the actual link to your GitHub repo
    st.link_button("View Project on GitHub", "https://github.com/mercuryafk/timeseriesanalysis")

# --- Main Dashboard ---
st.title("Time Series Analysis & Forecasting Dashboard")
st.markdown("A comparative analysis of ARIMA, SARIMA, Prophet, and LSTM models for Apple (AAPL) stock.")

# Load all pre-computed data into a dictionary
all_files = {
    "stock_data": load_data('apple_stock_data.csv'),
    "arima_forecast": load_data('arima_forecast.csv'),
    "sarima_forecast": load_data('sarima_forecast.csv'),
    "prophet_forecast": load_prophet_data('prophet_forecast.csv'),
    "lstm_forecast": load_data('lstm_forecast.csv'),
    "lstm_future_forecast": load_data('lstm_future_forecast.csv')
}

# Check if all files are loaded
if any(df is None for df in all_files.values()):
    st.error("One or more required CSV files are missing. Please run `compute_all_forecasts.py` first.")
else:
    # --- UI Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Historical Analysis", "ARIMA", "SARIMA", "Prophet", "LSTM"])

    with tab1:
        st.header("Historical Price Analysis for AAPL")
        fig = go.Figure(data=[go.Candlestick(x=all_files["stock_data"].index,
                                             open=all_files["stock_data"]['Open'],
                                             high=all_files["stock_data"]['High'],
                                             low=all_files["stock_data"]['Low'],
                                             close=all_files["stock_data"]['Close'],
                                             name='AAPL')])
        fig.add_trace(
            go.Scatter(x=all_files["stock_data"].index, y=all_files["stock_data"]['Close'].rolling(window=50).mean(),
                       line=dict(color='orange', width=1), name='50-Day MA'))
        fig.update_layout(title_text='AAPL Price History', xaxis_rangeslider_visible=False,
                          xaxis_rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The candlestick chart provides a detailed view of daily price movements. The 50-Day Moving Average (orange line) helps to smooth out short-term volatility and identify the underlying trend.
        """)
        st.subheader("Trading Volume")
        st.bar_chart(all_files["stock_data"]['Volume'])
        st.markdown("""
        **Insight:** Spikes in trading volume often coincide with major news events or earnings reports and can be correlated with significant price changes.
        """)

    with tab2:
        st.header("ARIMA Model Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=all_files["stock_data"].index[-200:], y=all_files["stock_data"]['Close'][-200:],
                                 name='Historical Price'))
        fig.add_trace(go.Scatter(x=all_files["arima_forecast"].index, y=all_files["arima_forecast"].iloc[:, 0],
                                 name='ARIMA Forecast', line={'dash': 'dash'}))
        fig.update_layout(title_text='ARIMA Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The ARIMA model serves as a statistical baseline. Its linear forecast suggests that based on historical autocorrelation alone, the stock follows a "random walk" where the best statistical prediction is a continuation of the recent trend.
        """)

    with tab3:
        st.header("SARIMA Model Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=all_files["stock_data"].index[-200:], y=all_files["stock_data"]['Close'][-200:],
                                 name='Historical Price'))
        fig.add_trace(go.Scatter(x=all_files["sarima_forecast"].index, y=all_files["sarima_forecast"].iloc[:, 0],
                                 name='SARIMA Forecast', line={'dash': 'dash'}))
        fig.update_layout(title_text='SARIMA (Quarterly) Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** The SARIMA model extends ARIMA by searching for seasonal patterns. The resulting linear forecast indicates that the model did not find strong, predictable quarterly patterns in the stock data, a common outcome for financial time series.
        """)

    with tab4:
        st.header("Prophet Model Forecast")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=all_files["stock_data"].index, y=all_files["stock_data"]['Close'], name='Actual Price'))
        fig.add_trace(go.Scatter(x=all_files["prophet_forecast"]['ds'], y=all_files["prophet_forecast"]['yhat'],
                                 name='Prophet Forecast'))
        fig.update_layout(title_text='Prophet Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight:** Prophet is an automated forecasting model that excels at decomposition. Its main value comes from identifying underlying weekly and yearly seasonalities, providing deeper insights into the stock's typical behavior throughout the year.
        """)

    with tab5:
        st.header("LSTM Model")
        st.subheader("Evaluation: Prediction vs. Actual (Test Set)")
        comparison_df = pd.DataFrame({
            'Actual Price': all_files["lstm_forecast"]['Close'],
            'LSTM Prediction': all_files["lstm_forecast"]['LSTM_Prediction']
        }).dropna()
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=comparison_df.index, y=comparison_df['Actual Price'], name='Actual Test Price'))
        fig_eval.add_trace(
            go.Scatter(x=comparison_df.index, y=comparison_df['LSTM Prediction'], name='LSTM Predictions'))
        fig_eval.update_layout(title_text='LSTM Prediction vs. Actual Price', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_eval, use_container_width=True)
        st.markdown("""
        **Insight:** This chart demonstrates the high accuracy of the enhanced LSTM model. The predictions on the unseen test data closely track the actual prices, proving its ability to learn complex, non-linear patterns.
        """)

        st.subheader("3-Month Future Forecast")
        fig_future = go.Figure()
        fig_future.add_trace(
            go.Scatter(x=all_files["stock_data"].index[-200:], y=all_files["stock_data"]['Close'][-200:],
                       name='Historical Price'))
        fig_future.add_trace(
            go.Scatter(x=all_files["lstm_future_forecast"].index, y=all_files["lstm_future_forecast"].iloc[:, 0],
                       name='Future Forecast', line={'dash': 'dash', 'color': 'red'}))
        fig_future.update_layout(title_text='Enhanced LSTM Future Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_future, use_container_width=True)
        st.markdown("""
        **Insight:** This chart shows the LSTM's projection into the unknown future. It's generated iteratively and demonstrates the model's extrapolation of the most recent trend. This type of long-range forecast is speculative and highlights the model's limitations.

        """)

