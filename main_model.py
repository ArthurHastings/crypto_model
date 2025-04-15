from imports import *


with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

url_api = "http://localhost:5002/invocations"

api_key = "RY3ZZPUFQJDVXAN0"
stock_symbol = "AAPL"
url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_symbol}&apikey={api_key}&limit=5000&sort=latest"


def api_call(sentence_pad_list):

    sentence_pad_list = np.array(sentence_pad_list)

    sentence_pad_list = np.squeeze(sentence_pad_list)

    data = {
        "instances": sentence_pad_list.tolist()
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url_api, data=json.dumps(data), headers=headers)
    response_json = response.json()
    predictions = response_json.get("predictions", [[]])
    return predictions


if __name__ == "__main__":
    # print(f"Now: {datetime.now()}")
    # print(f"Yesterday: {datetime.now() - timedelta(days=1)}")

    response_news = requests.get(url)
    news_data = response_news.json()

    articles = news_data.get("feed", [])

    date_limit_high = datetime.now() - timedelta(days=25)
    date_limit_low = datetime.now() - timedelta(days=30)

    filtered_articles = []
    for article in articles:
        article_time = datetime.strptime(article["time_published"], "%Y%m%dT%H%M%S")
        if article_time >= date_limit_low and article_time <= date_limit_high:
            filtered_articles.append(article)

    nr = 1
    print(f"Total articles fetched: {len(filtered_articles)}")
    title_list = []
    summary_list = []
    for article in filtered_articles:
        print(f"Article {nr}")
        title_list.append(article.get("title", "No title available"))
        print("Title:", article.get("title", "No title available"))
        summary_list.append(article.get("summary", "No summary available"))
        print("Summary:", article.get("summary", "No summary available"))
        print("Published:", article.get("time_published", "Unknown date"))
        print("----")
        nr += 1

    max_length = 0
    for i in range(len(title_list)):
        title_len = len(title_list[i])
        summary_len = len(summary_list[i])
        
        max_length = max(max_length, title_len, summary_len)

    # print(f"Max length is: {max_length}")
    sentence_pad_list = []
    for i in range(len(title_list)):
        sentence = title_list[i]
        sentence_clean = preprocess_text(sentence)
        sentence_seq = tokenizer.texts_to_sequences([sentence_clean])
        sentence_pad = pad_sequences(sentence_seq, maxlen=max_length, padding='post')
        sentence_pad_list.append(sentence_pad)

    response_sentiment = api_call(sentence_pad_list)

    for i, sentiment in enumerate(response_sentiment):
        try:
            predicted_label = np.argmax(sentiment)
            label_reverse_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            print(f"Sentiment {i}: {sentiment}")
            print(f"Predicted Sentiment: {label_reverse_mapping[predicted_label]} {sentiment[predicted_label] * 100:.2f}%")
        except:
            print("Empty sentiment list!")

