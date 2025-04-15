import pandas as pd
from get_news_sentences import period

file_path_headline = f"headline_sentiments{period}d.csv"
file_path_summary = f"summary_sentiments{period}d.csv"
df_headline = pd.read_csv(file_path_headline)
df_summary = pd.read_csv(file_path_summary)

df_headline = df_headline[pd.to_datetime(df_headline['Date'], errors='coerce').notna()]
df_summary = df_summary[pd.to_datetime(df_summary['Date'], errors='coerce').notna()]

df_headline['Date'] = pd.to_datetime(df_headline['Date'])
df_summary['Date'] = pd.to_datetime(df_summary['Date'])

df_summary = df_summary[df_summary["Headline"] != "Looking for stock market analysis and research with proves results? Zacks.com offers in-depth financial research with over 30years of proven results."]

df_summary.to_csv(f"test_cox.csv", index=False)

date_range = pd.date_range(end=df_headline['Date'].max(), periods=360)

full_dates_df = pd.DataFrame({'Date': date_range})

df_full_headline = full_dates_df.merge(df_headline, on='Date', how='left')
df_full_summary = full_dates_df.merge(df_summary, on='Date', how='left')

df_full_headline['Headline'].fillna('No article', inplace=True)
df_full_headline['Negative'].fillna(0, inplace=True)
df_full_headline['Neutral'].fillna(1, inplace=True)
df_full_headline['Positive'].fillna(0, inplace=True)
df_full_summary['Headline'].fillna('No article', inplace=True)
df_full_summary['Negative'].fillna(0, inplace=True)
df_full_summary['Neutral'].fillna(1, inplace=True)
df_full_summary['Positive'].fillna(0, inplace=True)

df_full_headline.to_csv(f"filled_headline_sentiments{period}d.csv", index=False)
df_full_summary.to_csv(f"filled_summary_sentiments{period}d.csv", index=False)
# Looking for stock market analysis and research with proves results? Zacks.com offers in-depth financial research with over 30years of proven results.