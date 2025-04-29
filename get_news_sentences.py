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

api_key = "cvpq5t9r01qve7iqiis0cvpq5t9r01qve7iqiisg"
api_sentiment_model = "http://localhost:5002/invocations"
stock_symbol = "AAPL"
headline_dict = {"Date": [], "Sentence": []}
summary_dict = {"Date": [], "Sentence": []}

period = 360

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def api_call_batch(sentence_pad_list_batch):
    batch_array = np.array(sentence_pad_list_batch)
    batch_array = np.squeeze(batch_array)
    
    data = {"instances": batch_array.tolist()}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(api_sentiment_model, data=json.dumps(data), headers=headers)
    try:
        response_json = response.json()
    except json.decoder.JSONDecodeError as e:
        print("JSONDecodeError on batch. Check response content.", e)
        raise
    return response_json.get("predictions", [])

def api_call(sentence_pad_dict: dict, batch_size=200):
    all_predictions = {"Date": [], "Sentence": []}
    total = len(sentence_pad_dict["Sentence"])
    
    reversed_sentence_pad_dict = {"Sentence": sentence_pad_dict["Sentence"][::-1], "Date": sentence_pad_dict["Date"][::-1],}

    for i in range(0, total, batch_size):
        batch_sentences = reversed_sentence_pad_dict["Sentence"][i: i + batch_size]
        batch_dates = reversed_sentence_pad_dict["Date"][i: i + batch_size]
        print(f"Processing batch {i} to {i + len(batch_sentences)} of {total}")
        preds = api_call_batch(batch_sentences)
        all_predictions["Sentence"].extend(preds)
        all_predictions["Date"].extend(batch_dates)

    return all_predictions


def get_news(days: int):
    for i in range(days // 10):
        date_from = (datetime.now() - timedelta(days=10 * (i + 1))).strftime("%Y-%m-%d")
        date_to   = (datetime.now() - timedelta(days=10 * i)).strftime("%Y-%m-%d")
        url = f"https://finnhub.io/api/v1/company-news?symbol={stock_symbol}&from={date_from}&to={date_to}&token={api_key}"
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
    return max_length

if __name__ == "__main__":
    max_length = get_news(period)

    # print(headline_dict["Sentence"][len(headline_dict["Sentence"]) - 50:])
    # print("-----------" * 10)
    # print(summary_dict["Sentence"][len(headline_dict["Sentence"]) - 50:])

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
    
    response_headline_sentiment = api_call(sentence_pad_dict_headline, batch_size=200)
    df_headlines = pd.DataFrame({
        "Date": response_headline_sentiment["Date"],
        "Headline": headline_dict["Sentence"][::-1],        
        "Negative": [pred[0] for pred in response_headline_sentiment["Sentence"]],
        "Neutral": [pred[1] for pred in response_headline_sentiment["Sentence"]],
        "Positive": [pred[2] for pred in response_headline_sentiment["Sentence"]]
    })
    df_headlines.to_csv(f"headline_sentiments{period}d.csv", index=False)
    print("Headline predictions saved to headline_sentiments.csv")

    response_summary_sentiment = api_call(sentence_pad_dict_summary, batch_size=200)
    df_summary = pd.DataFrame({
        "Date": response_summary_sentiment["Date"],
        "Headline": summary_dict["Sentence"][::-1],        
        "Negative": [pred[0] for pred in response_summary_sentiment["Sentence"]],
        "Neutral": [pred[1] for pred in response_summary_sentiment["Sentence"]],
        "Positive": [pred[2] for pred in response_summary_sentiment["Sentence"]]
    })
    df_summary.to_csv(f"summary_sentiments{period}d.csv", index=False)
    print("Summary predictions saved to headline_sentiments.csv")