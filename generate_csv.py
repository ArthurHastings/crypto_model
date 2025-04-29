import pandas as pd
import yfinance as yf
import os

period = 360
stock_symbol = "AAPL"

headline_df = pd.read_csv(f"filled_headline_sentiments{period}d.csv")
summary_df = pd.read_csv(f"filled_summary_sentiments{period}d.csv")

headline_daily_avg = headline_df.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()
summary_daily_avg = summary_df.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()

headline_daily_avg.columns = ['Date', 'Headline_Negative', 'Headline_Neutral', 'Headline_Positive']
summary_daily_avg.columns = ['Date', 'Summary_Negative', 'Summary_Neutral', 'Summary_Positive']

combined = pd.merge(headline_daily_avg, summary_daily_avg, on="Date", how="outer")
final_avg_sentiment = pd.DataFrame()
final_avg_sentiment['Date'] = combined['Date']
final_avg_sentiment['Negative'] = (combined['Headline_Negative'] + combined['Summary_Negative']) / 2
final_avg_sentiment['Neutral'] = (combined['Headline_Neutral'] + combined['Summary_Neutral']) / 2
final_avg_sentiment['Positive'] = (combined['Headline_Positive'] + combined['Summary_Positive']) / 2

df_price = yf.download(stock_symbol, period=f"{period}d", interval="1d").reset_index()
df_price.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

df_price['Target'] = (df_price['Close'].shift(-1) > df_price['Close']).astype(int)

df_price["Date"] = pd.to_datetime(df_price["Date"])
final_avg_sentiment["Date"] = pd.to_datetime(final_avg_sentiment["Date"])
min_sentiment_date = final_avg_sentiment["Date"].min()
df_price = df_price[df_price["Date"] >= min_sentiment_date].reset_index(drop=True)

df_price = pd.merge(df_price, final_avg_sentiment, on='Date', how='left')
df_price['Negative'].fillna(0, inplace=True)
df_price['Neutral'].fillna(0, inplace=True)
df_price['Positive'].fillna(0, inplace=True)

df_price.to_csv(f"apple_price_sentiment_{period}d.csv", index=False)
print(f"Saved final dataset: apple_price_sentiment_{period}d.csv")

intermediate_files = [
    f"filled_headline_sentiments{period}d.csv",
    f"filled_summary_sentiments{period}d.csv"
]

for file in intermediate_files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except FileNotFoundError:
        print(f"File not found for deletion: {file}")
