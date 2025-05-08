import quandl

API_KEY = 'X3-tTgpfWyNtfe6u9rHN'

# Example: Fetch Apple stock price data
data = quandl.get("WIKI/AAPL", api_key=API_KEY)

# Print the first few rows
print(data.head())
