# -*- coding: utf-8 -*-
"""
Final Offline Model Computation Code

This code trains all finalized forecasting models (ARIMA, SARIMA, Prophet, and
an enhanced LSTM) and saves their predictions to CSV files. This is the single
source for all pre-computed results used by the dashboard.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

warnings.filterwarnings("ignore")


# --- 1. Data Loading and Cleaning ---
def load_and_clean_data(file_path):
    """Loads and robustly cleans the stock data from a CSV file."""
    print(f"Loading and cleaning data from {file_path}...")
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
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{file_path}' was not found.")
        return None


# --- 2. Model Computation Functions ---

def compute_arima_forecast(data, filename='arima_forecast.csv'):
    """Trains an ARIMA model and saves its forecast."""
    print("\n--- Training ARIMA model ---")
    try:
        arima_data = data['Close'].asfreq('D').fillna(method='ffill')
        model = ARIMA(arima_data, order=(3, 1, 3))
        results = model.fit()
        forecast = results.get_forecast(steps=365).predicted_mean
        forecast.to_csv(filename, header=['ARIMA_Forecast'])
        print(f"ARIMA forecast saved to '{filename}'")
    except Exception as e:
        print(f"ARIMA model failed: {e}")


def compute_sarima_forecast(data, filename='sarima_forecast.csv'):
    """Trains a SARIMA model (quarterly) and saves its forecast."""
    print("\n--- Training SARIMA model ---")
    try:
        sarima_data = data['Close'].asfreq('D').fillna(method='ffill')
        model = SARIMAX(sarima_data, order=(3, 1, 3), seasonal_order=(1, 1, 1, 63))
        results = model.fit(disp=False)
        forecast = results.get_forecast(steps=365).predicted_mean
        forecast.to_csv(filename, header=['SARIMA_Forecast'])
        print(f"SARIMA forecast saved to '{filename}'")
    except Exception as e:
        print(f"SARIMA model failed: {e}")


def compute_prophet_forecast(data, filename='prophet_forecast.csv'):
    """Trains a Prophet model and saves its forecast."""
    print("\n--- Training Prophet model ---")
    try:
        df_prophet = data[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(filename, index=False)
        print(f"Prophet forecast saved to '{filename}'")
    except Exception as e:
        print(f"Prophet model failed: {e}")


def compute_lstm_forecast(data, eval_filename='lstm_forecast.csv', future_filename='lstm_future_forecast.csv'):
    """Trains an enhanced LSTM model, saves its test-set evaluation, and generates a future forecast."""
    print("\n--- Training Enhanced LSTM model (this is the slowest part) ---")
    try:
        # --- Part 1: Prepare Data ---
        dataset = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # --- Part 2: Build and Train Enhanced Model for Evaluation ---
        training_data_len = int(np.ceil(len(scaled_data) * 0.8))
        train_data = scaled_data[0:training_data_len]
        test_data = scaled_data[training_data_len - 60:]

        X_train, y_train = [], []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Fitting LSTM model (50 epochs)... This will take several minutes.")
        model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)

        # --- Part 3: Evaluate on Test Set and Save ---
        X_test, y_test = [], dataset[training_data_len:]
        for i in range(60, len(test_data)):
            X_test.append(test_data[i - 60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        valid = data[training_data_len:].copy()
        valid['LSTM_Prediction'] = predictions
        valid.to_csv(eval_filename)
        print(f"LSTM test predictions saved to '{eval_filename}'")

        # --- Part 4: Generate and Save 3-Month Future Forecast ---
        print("\nGenerating new 3-month future forecast...")
        last_60_days = scaled_data[-60:]
        future_predictions = []
        for _ in range(90):  # 3 months
            X_future = np.reshape(last_60_days, (1, 60, 1))
            predicted_price_scaled = model.predict(X_future)
            future_predictions.append(predicted_price_scaled[0, 0])
            last_60_days = np.append(last_60_days[1:], predicted_price_scaled, axis=0)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=90)
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['LSTM_Future_Forecast'])
        future_df.to_csv(future_filename)
        print(f"LSTM future forecast saved to '{future_filename}'")

    except Exception as e:
        print(f"LSTM model failed: {e}")


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    source_file = 'apple_stock_data.csv'
    stock_data = load_and_clean_data(source_file)

    if stock_data is not None:
        compute_arima_forecast(stock_data)
        compute_sarima_forecast(stock_data)
        compute_prophet_forecast(stock_data)
        compute_lstm_forecast(stock_data)
        print("\n✅ All models have been trained and forecasts have been exported to CSV files.")