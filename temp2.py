import pandas as pd
from get_news_sentences import period
import os

headline_daily_avg = pd.read_csv(f"headline_daily_sentiment_avg{period}d.csv")
summary_daily_avg = pd.read_csv(f"summary_daily_sentiment_avg{period}d.csv")

headline_daily_avg.columns = ['Date', 'Headline_Negative', 'Headline_Neutral', 'Headline_Positive']
summary_daily_avg.columns = ['Date', 'Summary_Negative', 'Summary_Neutral', 'Summary_Positive']

combined = pd.merge(headline_daily_avg, summary_daily_avg, on="Date", how="outer")

final_avg_sentiment = pd.DataFrame()
final_avg_sentiment['Date'] = combined['Date']
final_avg_sentiment['Negative'] = (combined['Headline_Negative'] + combined['Summary_Negative']) / 2
final_avg_sentiment['Neutral'] = (combined['Headline_Neutral'] + combined['Summary_Neutral']) / 2
final_avg_sentiment['Positive'] = (combined['Headline_Positive'] + combined['Summary_Positive']) / 2

final_avg_sentiment.to_csv(f"final_daily_avg_sentiment{period}d.csv", index=False)
try:
    os.remove(f"headline_daily_sentiment_avg{period}d.csv")
except FileNotFoundError:
    print(f"File headline_daily_sentiment_avg{period}d.csv not found.")

try:
    os.remove(f"summary_daily_sentiment_avg{period}d.csv")
except FileNotFoundError:
    print(f"File summary_daily_sentiment_avg{period}d.csv not found.")
