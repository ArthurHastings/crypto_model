import pandas as pd 
import os 

stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD", "BA", "JPM", "DIS", "V", "NKE",
                "PYPL", "KO", "PEP", "PFE", "MRK", "CVX", "XOM", "MCD", "WMT", "ORCL", "IBM", "UNH", "COST", "BAC", "SNOW"]

path = "stock_csvs"

for stock_symbol in stock_list:

    csv_path = os.path.join(path, f"{stock_symbol}_price_sentiment.csv")
    df = pd.read_csv(csv_path)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.to_csv(csv_path, index=False)