import pandas as pd
import os


def clean_files(df_headline_received, df_summary_received, stock_symbol):
    df_headline = df_headline_received
    df_summary = df_summary_received

    df_headline = df_headline[pd.to_datetime(df_headline['Date'], errors='coerce').notna()]
    df_summary = df_summary[pd.to_datetime(df_summary['Date'], errors='coerce').notna()]

    df_headline['Date'] = pd.to_datetime(df_headline['Date'])
    df_summary['Date'] = pd.to_datetime(df_summary['Date'])

    df_summary = df_summary[df_summary["Headline"] != "Looking for stock market analysis and research with proves results? Zacks.com offers in-depth financial research with over 30years of proven results."]

    date_range = pd.date_range(end=df_headline['Date'].max(), periods=360)

    full_dates_df = pd.DataFrame({'Date': date_range})

    df_full_headline = full_dates_df.merge(df_headline, on='Date', how='left')
    df_full_summary = full_dates_df.merge(df_summary, on='Date', how='left')

    df_full_headline['Headline'] = df_full_headline['Headline'].fillna('No article')
    df_full_headline['Negative'] = df_full_headline['Negative'].fillna(0)
    df_full_headline['Neutral'] = df_full_headline['Neutral'].fillna(1)
    df_full_headline['Positive'] = df_full_headline['Positive'].fillna(0)

    df_full_summary['Headline'] = df_full_summary['Headline'].fillna('No article')
    df_full_summary['Negative'] = df_full_summary['Negative'].fillna(0)
    df_full_summary['Neutral'] = df_full_summary['Neutral'].fillna(1)
    df_full_summary['Positive'] = df_full_summary['Positive'].fillna(0)


    df_full_headline.to_csv(f"{stock_symbol}_headline_cleaned.csv", index=False)
    df_full_summary.to_csv(f"{stock_symbol}_summary_cleaned.csv", index=False)
    
    return f"{stock_symbol}_headline_cleaned.csv", f"{stock_symbol}_summary_cleaned.csv"