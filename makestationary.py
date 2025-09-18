import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load and clean the Apple stock data
file_path = 'apple_stock_data.csv'
df = pd.read_csv(file_path, index_col=0, parse_dates=True)
close_prices_numeric = pd.to_numeric(df['Close'], errors='coerce')
time_series = close_prices_numeric.dropna()

# --- Step 1: Calculate the first difference ---
diff_series = time_series.diff().dropna()

# --- Step 2: Plot the original and differenced series ---
plt.style.use('fivethirtyeight')

# Plot original series
plt.figure(figsize=(12, 6))
plt.title('Original Apple Stock Price (Non-Stationary)')
plt.plot(time_series, label='Original Close Price')
plt.legend()
plt.show()

# Plot differenced series
plt.figure(figsize=(12, 6))
plt.title('Differenced Apple Stock Price (Stationary)')
plt.plot(diff_series, label='First Difference', color='orange')
plt.legend()
plt.show()

# --- Step 3: Run the ADF test on the differenced series ---
print("\n--- Running ADF Test on the Differenced Series ---")
adf_result = adfuller(diff_series)

print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print("\n--- Conclusion ---")
if adf_result[1] > 0.05:
    print("The p-value is > 0.05. The series is still NON-STATIONARY.")
else:
    print("The p-value is <= 0.05. The series is now STATIONARY.")