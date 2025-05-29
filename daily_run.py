import os
import nltk
import time
from datetime import datetime, timedelta

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

from imports import *
from get_news_sentences import api_key, get_news, preprocess_sentences, api_call, api_call_batch, create_dataset
from clean_headline_sentiment import clean_files
from generate_csvs import generate_csv
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed(username='SOLOMON_ROCKS', password='zazacox1234567!')
api_sentiment_model = os.getenv("API_SENTIMENT_MODEL", "http://localhost:5002/invocations")
stock_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD", "BA", "JPM", "DIS", "V", "NKE",
                "PYPL", "KO", "PEP", "PFE", "MRK", "CVX", "XOM", "MCD", "WMT", "ORCL", "IBM", "UNH", "COST", "BAC", "SNOW"]

csv_folder = "stock_csvs"
os.makedirs(csv_folder, exist_ok=True)

for stock_symbol in stock_list:
    print("Waiting 5 seconds...")
    time.sleep(5)
    try:
        print(f"\n===== Processing {stock_symbol} =====")

        csv_path = os.path.join(csv_folder, f"{stock_symbol}_price_sentiment.csv")
        if not os.path.exists(csv_path):
            print(f"CSV for {stock_symbol} not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])

        last_date = df["Date"].max()
        today = datetime.now()
        nr_days = (today - last_date).days

        if nr_days < 1:
            print(f"No new data needed for {stock_symbol}.")
            continue

        print(f"Updating {stock_symbol} from {last_date.date()} to {today.date()} ({nr_days} days)")

        max_length, headline_dict, summary_dict = get_news(nr_days, stock_symbol, api_key)

        sentence_pad_dict_headline, sentence_pad_dict_summary = preprocess_sentences(max_length, headline_dict, summary_dict)

        response_headline_sentiment = api_call(sentence_pad_dict_headline, api_sentiment_model, batch_size=200)
        df_headlines = create_dataset(response_headline_sentiment, headline_dict)
        
        response_summary_sentiment = api_call(sentence_pad_dict_summary, api_sentiment_model, batch_size=200)
        df_summary = create_dataset(response_summary_sentiment, summary_dict)

        temp_df_name = generate_csv(df_headlines, df_summary, nr_days, stock_symbol, tv)
        print("ZAZA" * 40)
        new_df = pd.read_csv(temp_df_name)
        os.remove(temp_df_name)
        print(f"Deleted temp file: {temp_df_name}")

        new_df["Date"] = pd.to_datetime(new_df["Date"])
        start_date = last_date + timedelta(days=1)
        filtered_df = new_df[new_df["Date"] >= start_date].copy()
        filtered_df.reset_index(drop=True, inplace=True)

        updated_df = pd.concat([df, filtered_df], ignore_index=True)
        updated_df.drop_duplicates(subset="Date", keep="last", inplace=True)
        updated_df["Date"] = updated_df["Date"].dt.strftime("%Y-%m-%d")

        updated_df.to_csv(csv_path, index=False)
        print(f"✅ Updated {csv_path} with {len(filtered_df)} new rows.")
    except Exception as e:
        for file in [f"{stock_symbol}_headline_cleaned.csv", f"{stock_symbol}_summary_cleaned.csv"]:
            try:
                os.remove(file)
                print(f"Deleted: {file}")
            except FileNotFoundError:
                print(f"File not found for deletion: {file}")
        print(f"❌ Failed to update {stock_symbol}: {e}")

# 2xlVy7FtVj4ZLNBV7WpFzW9YM-LX_7SZybH6WhTSrxB5zH91P6