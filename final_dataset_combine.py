import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

period = 360
stock_symbol = "AAPL"

if __name__ == "__main__":
    df_price = yf.download(stock_symbol, period=f"{period}d", interval="1d")

    df_price = df_price.reset_index()
    df_price.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    df_price['Target'] = (df_price['Close'].shift(-1) > df_price['Close']).astype(int)

    df_sentiment_headline = pd.read_csv(f"filled_headline_sentiments{period}d.csv")
    df_sentiment_summary = pd.read_csv(f"filled_summary_sentiments{period}d.csv")

    min_sentiment_date = df_sentiment_headline["Date"].min()
    df_price = df_price[df_price["Date"] >= min_sentiment_date].reset_index(drop=True)

    df_price["Sentiment headline"] = 0
    df_price["Sentiment summary"] = 0

    def compute_sentiment_headline(date):
        matching_rows = df_sentiment_headline[df_sentiment_headline["Date"] == date]
        
        if not matching_rows.empty:
            dominant_sentiments = matching_rows[["Negative", "Neutral", "Positive"]].idxmax(axis=1)
            
            sentiment_scores = dominant_sentiments.map({"Negative": -1, "Neutral": 0, "Positive": 1})
            
            return sentiment_scores.mean()
        
        return 0.0

    def compute_sentiment_summary(date):
        matching_rows = df_sentiment_summary[df_sentiment_summary["Date"] == date]
        
        if not matching_rows.empty:
            dominant_sentiments = matching_rows[["Negative", "Neutral", "Positive"]].idxmax(axis=1)
            
            sentiment_scores = dominant_sentiments.map({"Negative": -1, "Neutral": 0, "Positive": 1})
            
            return sentiment_scores.mean()
        
        return 0.0

    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_sentiment_headline["Date"] = pd.to_datetime(df_sentiment_headline["Date"])
    df_sentiment_summary["Date"] = pd.to_datetime(df_sentiment_summary["Date"])

    df_price["Sentiment headline"] = df_price["Date"].apply(compute_sentiment_headline)
    df_price["Sentiment summary"] = df_price["Date"].apply(compute_sentiment_summary)

    df_price.to_csv(f"apple_price_sentiment_{period}d.csv", index=False)