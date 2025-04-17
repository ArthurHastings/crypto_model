import pandas as pd
from get_news_sentences import period
import os

headline_df = pd.read_csv(f"filled_headline_sentiments{period}d.csv")
summary_df = pd.read_csv(f"filled_summary_sentiments{period}d.csv")

headline_daily_avg = headline_df.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()
summary_daily_avg = summary_df.groupby("Date")[["Negative", "Neutral", "Positive"]].mean().reset_index()

headline_output_path = f"headline_daily_sentiment_avg{period}d.csv"
summary_output_path = f"summary_daily_sentiment_avg{period}d.csv"

headline_daily_avg.to_csv(headline_output_path, index=False)
summary_daily_avg.to_csv(summary_output_path, index=False)

try:
    os.remove(f"filled_headline_sentiments{period}d.csv")
except FileNotFoundError:
    print(f"File filled_headline_sentiments{period}d.csv not found.")

try:
    os.remove(f"filled_summary_sentiments{period}d.csv")
except FileNotFoundError:
    print(f"File filled_summary_sentiments{period}d.csv not found.")