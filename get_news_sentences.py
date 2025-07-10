import os
import nltk

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

from imports import *
from generate_csvs import generate_csv
from clean_headline_sentiment import clean_files
from tvDatafeed import TvDatafeed, Interval
# --------------------------------------------------------------------------------------------------------------------------------

api_key = "cvpq5t9r01qve7iqiis0cvpq5t9r01qve7iqiisg"
# api_sentiment_model = "http://localhost:5002/invocations"
tv = TvDatafeed(username='SOLOMON_ROCKS', password='Tempword12345!')

period = 360

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def api_call_batch(sentence_pad_list_batch, api_sentiment_model):
    batch_array = np.array(sentence_pad_list_batch)
    batch_array = np.squeeze(batch_array)
    
    data = {"instances": batch_array.tolist()}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(api_sentiment_model, data=json.dumps(data), headers=headers)
    # print("Response status code:", response.status_code)
    # print("Response text:", response.text[:500])

    try:
        response_json = response.json()
    except json.decoder.JSONDecodeError as e:
        print("JSONDecodeError on batch. Check response content.", e)
        raise
    return response_json.get("predictions", [])

def api_call(sentence_pad_dict: dict, api_sentiment_model, batch_size=200):
    all_predictions = {"Date": [], "Sentence": []}
    total = len(sentence_pad_dict["Sentence"])
    
    reversed_sentence_pad_dict = {"Sentence": sentence_pad_dict["Sentence"][::-1], "Date": sentence_pad_dict["Date"][::-1],}

    for i in range(0, total, batch_size):
        batch_sentences = reversed_sentence_pad_dict["Sentence"][i: i + batch_size]
        batch_dates = reversed_sentence_pad_dict["Date"][i: i + batch_size]
        print(f"Processing batch {i} to {i + len(batch_sentences)} of {total}")
        preds = api_call_batch(batch_sentences, api_sentiment_model)
        all_predictions["Sentence"].extend(preds)
        all_predictions["Date"].extend(batch_dates)

    return all_predictions


def get_news(days: int, symbol, api_key):
    headline_dict = {"Date": [], "Sentence": []}
    summary_dict = {"Date": [], "Sentence": []}
    for i in range(0, days, 10):
        days_to_subtract_start = i
        days_to_subtract_end = min(i + 10, days)
        date_from = (datetime.now() - timedelta(days=days_to_subtract_end)).strftime("%Y-%m-%d")
        date_to   = (datetime.now() - timedelta(days=days_to_subtract_start)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={date_from}&to={date_to}&token={api_key}"
        response_news = requests.get(url)
        news_data = response_news.json()
        print(f"Number of articles found between {date_from} to {date_to}: {len(news_data)}")
        if not news_data:
            print("No articles found in the given date range.")
        else:
            for article in news_data:
                if not isinstance(article, dict):
                    print(f"Unexpected article format: {article}")
                    continue
                timestamp = article.get('datetime')
                if timestamp and isinstance(timestamp, (int, float)) and timestamp > 0:
                    date_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d')
                else:
                    date_str = "Unknown Date"

                headline_dict["Sentence"].append(article.get('headline', 'No title'))
                headline_dict["Date"].append(date_str)
                summary_dict["Sentence"].append(article.get('summary', 'No summary'))
                summary_dict["Date"].append(date_str)
    print(f"Nr of headlines: {len(headline_dict['Date'])}")
    print(f"Nr of summaries: {len(summary_dict['Date'])}")
    
    max_length = 0
    for i in range(len(headline_dict["Sentence"])):
        title_len = len(headline_dict["Sentence"][i])
        summary_len = len(summary_dict["Sentence"][i])
        if title_len > max_length:
            max_length = title_len
        if summary_len > max_length:
            max_length = summary_len
    return max_length, headline_dict, summary_dict

def preprocess_sentences(max_length, headline_dict, summary_dict):
    sentence_pad_dict_headline = {"Date": [], "Sentence": []}
    sentence_pad_dict_summary = {"Date": [], "Sentence": []}
    for i in range(len(headline_dict["Sentence"])):
        sentence_headline = headline_dict["Sentence"][i]
        sentence_summary = summary_dict["Sentence"][i]
        sentence_clean_headline = preprocess_text(sentence_headline)
        sentence_clean_summary = preprocess_text(sentence_summary)
        # print(headline_dict["Date"][i], sentence_clean_headline)
        sentence_seq_headline = tokenizer.texts_to_sequences([sentence_clean_headline])
        sentence_seq_summary = tokenizer.texts_to_sequences([sentence_clean_summary])
        sentence_pad_headline = pad_sequences(sentence_seq_headline, maxlen=max_length, padding='post')
        sentence_pad_summary = pad_sequences(sentence_seq_summary, maxlen=max_length, padding='post')
        sentence_pad_dict_headline["Sentence"].append(sentence_pad_headline)
        sentence_pad_dict_summary["Sentence"].append(sentence_pad_summary)
        sentence_pad_dict_headline["Date"].append(headline_dict["Date"][i])
        sentence_pad_dict_summary["Date"].append(headline_dict["Date"][i])

    print("Total headlines to process:", len(sentence_pad_dict_headline["Date"]))
    return sentence_pad_dict_headline, sentence_pad_dict_summary

def create_dataset(response_sentiment, original_dict):
    df = pd.DataFrame({
        "Date": response_sentiment["Date"],
        "Headline": original_dict["Sentence"][::-1],        
        "Negative": [pred[0] for pred in response_sentiment["Sentence"]],
        "Neutral": [pred[1] for pred in response_sentiment["Sentence"]],
        "Positive": [pred[2] for pred in response_sentiment["Sentence"]]
    })
    return df

def process_stock(symbol, api_sentiment_model):
    global stock_symbol
    stock_symbol = symbol

    print(f"\n--- Processing {stock_symbol} ---")
    max_length, headline_dict, summary_dict = get_news(period, symbol, api_key)

    sentence_pad_dict_headline, sentence_pad_dict_summary = preprocess_sentences(max_length, headline_dict, summary_dict)

    print(f"Using API URL: {api_sentiment_model}")
    response_headline_sentiment = api_call(sentence_pad_dict_headline, api_sentiment_model,  batch_size=200)
    df_headlines = create_dataset(response_headline_sentiment, headline_dict)

    response_summary_sentiment = api_call(sentence_pad_dict_summary, api_sentiment_model, batch_size=200)
    df_summary = create_dataset(response_summary_sentiment, summary_dict)

    final_df_name = generate_csv(df_headlines, df_summary, period, stock_symbol, tv)

    new_df = pd.read_csv(final_df_name)
    try:
        os.remove(final_df_name)
        print(f"Deleted: {final_df_name}")
    except FileNotFoundError:
        print(f"File not found for deletion: {final_df_name}")

    new_df["Date"] = pd.to_datetime(new_df["Date"])
    new_df["Date"] = new_df["Date"].dt.strftime("%Y-%m-%d")

    out_name = f"{stock_symbol}_price_sentiment.csv"
    new_df.to_csv(out_name, index=False)
    print(f"Saved updated dataset with {len(new_df)} rows to: {out_name}")

if __name__ == "__main__":
    api_sentiment_model = os.getenv("API_SENTIMENT_MODEL", "http://localhost:5002/invocations")
    # stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "INTC", "AMD", "BA", "JPM", "DIS", "V", "NKE"] # original 15 stocks
    stock_symbols = ["PYPL", "KO", "PEP", "PFE", "MRK", "CVX", "XOM", "MCD", "WMT", "ORCL", "IBM", "UNH", "COST", "BAC", "SNOW"]
    for symbol in stock_symbols:
        process_stock(symbol, api_sentiment_model)
        print(f"Waiting 2 minutes to avoid rate limits before next stock...")
        time.sleep(120)
