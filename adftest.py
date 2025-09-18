import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Load the Apple stock data
file_path = 'apple_stock_data.csv'
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# --- DIAGNOSTIC STEP ---
# Let's look at the first few rows to see the extra header row
print("--- Data Preview (First 5 Rows) ---")
print(df.head())
print("-" * 40)

# --- CLEANING STEP ---
# The 'Close' column contains non-numeric text.
# We convert the column to numbers; 'coerce' turns any text that
# can't be converted (like 'AAPL') into an empty value (NaN).
close_prices_numeric = pd.to_numeric(df['Close'], errors='coerce')

# Now, we drop any of those empty NaN values to get a clean series of numbers.
time_series = close_prices_numeric.dropna()


print("\n--- Running ADF Test on Cleaned Data ---")

# Perform the ADF test on our now-clean, numeric data
adf_result = adfuller(time_series)

# Nicely format and print the results
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print('Critical Values:')
for key, value in adf_result[4].items():
    print(f'\t{key}: {value}')

# Print the conclusion
print("\n--- Conclusion ---")
if adf_result[1] > 0.05:
    print("The p-value is greater than 0.05, so we conclude the series is NON-STATIONARY.")
else:
    print("The p-value is less than or equal to 0.05, so we conclude the series is STATIONARY.")