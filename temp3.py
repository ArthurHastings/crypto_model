import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from get_news_sentences import period
import os

stock_symbol = "AAPL"

if __name__ == "__main__":
    df_price = yf.download(stock_symbol, period=f"{period}d", interval="1d")

    df_price = df_price.reset_index()
    df_price.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    df_price['Target'] = (df_price['Close'].shift(-1) > df_price['Close']).astype(int)

    df_sentiment = pd.read_csv(f"final_daily_avg_sentiment{period}d.csv")

    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])

    min_sentiment_date = df_sentiment["Date"].min()
    df_price = df_price[df_price["Date"] >= min_sentiment_date].reset_index(drop=True)

    df_price = pd.merge(df_price, df_sentiment[['Date', 'Negative', 'Neutral', 'Positive']], on='Date', how='left')

    df_price['Negative'].fillna(0, inplace=True)
    df_price['Neutral'].fillna(0, inplace=True)
    df_price['Positive'].fillna(0, inplace=True)

    df_price.to_csv(f"apple_price_sentiment_{period}d.csv", index=False)

    try:
        os.remove(f"final_daily_avg_sentiment{period}d.csv")
    except FileNotFoundError:
        print(f"File final_daily_avg_sentiment{period}d.csv not found.")
