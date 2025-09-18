import yfinance as yf
import pandas as pd
import sys

print("Attempting to fetch Apple Inc. (AAPL) stock data...")

# Ticker for Apple Inc.
ticker = "AAPL"
# Let's get data starting from 2015 to have a rich history
start_date = "2015-01-01"

try:
    df = yf.download(ticker, start=start_date)
except Exception as e:
    print(f"An error occurred during download: {e}")
    df = pd.DataFrame()

# --- VALIDATION STEP ---
if df.empty:
    print(f"Error: No data was downloaded for ticker '{ticker}'.")
    sys.exit()

print("\nSuccess! Data downloaded.")

# --- Step 1: Basic Preprocessing ---
df.index = pd.to_datetime(df.index)

# --- Step 2: Save the FULL Historical Data to CSV ---
file_path = 'apple_stock_data.csv'
df.to_csv(file_path)

print(f"\nSuccessfully saved the FULL data history for AAPL to '{file_path}'")
print("The file is now ready for Python analysis.")