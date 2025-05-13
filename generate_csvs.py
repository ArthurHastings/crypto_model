from symtable import Symbol
import pandas as pd
import yfinance as yf
import os
from clean_headline_sentiment import clean_files
from tvDatafeed import TvDatafeed, Interval

def generate_csv(headline_df, summary_df, nr_days, stock_symbol, tv):
    headline, summary = clean_files(headline_df, summary_df, stock_symbol)
    headline_df_cleaned = pd.read_csv(headline)
    summary_df_cleaned = pd.read_csv(summary)

    headline_daily_avg = headline_df_cleaned.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()
    summary_daily_avg = summary_df_cleaned.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()

    headline_daily_avg.columns = ['Date', 'Headline_Negative', 'Headline_Neutral', 'Headline_Positive']
    summary_daily_avg.columns = ['Date', 'Summary_Negative', 'Summary_Neutral', 'Summary_Positive']

    combined = pd.merge(headline_daily_avg, summary_daily_avg, on="Date", how="outer")
    final_avg_sentiment = pd.DataFrame()
    final_avg_sentiment['Date'] = combined['Date']
    final_avg_sentiment['Negative'] = (combined['Headline_Negative'] + combined['Summary_Negative']) / 2
    final_avg_sentiment['Neutral'] = (combined['Headline_Neutral'] + combined['Summary_Neutral']) / 2
    final_avg_sentiment['Positive'] = (combined['Headline_Positive'] + combined['Summary_Positive']) / 2
    
    if stock_symbol in ["BA", "JPM", "DIS", "V", "NKE"]:
        df_price = tv.get_hist(symbol=stock_symbol, exchange='NYSE', interval=Interval.in_daily, n_bars=nr_days)
    else:
        df_price = tv.get_hist(symbol=stock_symbol, exchange='NASDAQ', interval=Interval.in_daily, n_bars=nr_days)

    df_price = df_price.reset_index()

    df_price = df_price.rename(columns={
        'datetime': 'Date',
        'close': 'Close',
        'high': 'High',
        'low': 'Low',
        'open': 'Open',
        'volume': 'Volume'
    })


    df_price = df_price[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    df_price['Date'] = df_price['Date'].dt.strftime('%Y-%m-%d')

    print(df_price.head())
    df_price['Target'] = (df_price['Close'].shift(-1) > df_price['Close']).astype(int)

    df_price["Date"] = pd.to_datetime(df_price["Date"])

    final_avg_sentiment["Date"] = pd.to_datetime(final_avg_sentiment["Date"])
    min_sentiment_date = final_avg_sentiment["Date"].min()
    df_price = df_price[df_price["Date"] >= min_sentiment_date].reset_index(drop=True)

    df_price = pd.merge(df_price, final_avg_sentiment, on='Date', how='left')
    df_price['Negative'].fillna(0, inplace=True)
    df_price['Neutral'].fillna(0, inplace=True)
    df_price['Positive'].fillna(0, inplace=True)

    temp_file = f"{stock_symbol}_temp_price_sentiment.csv"
    df_price.to_csv(temp_file, index=False)
    print(f"Saved temporary dataset: {temp_file}")

    for file in [headline, summary]:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except FileNotFoundError:
            print(f"File not found for deletion: {file}")
    
    return temp_file
