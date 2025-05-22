import os
import nltk
import time

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    nltk.download('stopwords', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'wordnet')):
    nltk.download('wordnet', download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers', 'punkt')):
    nltk.download('punkt', download_dir=nltk_data_path)

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime


api_sentiment_model = os.getenv("MLFLOW_NGROK", "http://localhost:5003")
mlflow.set_tracking_uri(api_sentiment_model)

stock_models = {
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
    "NVDA":"NVDA",
    "META":"META",
    "NFLX":"NFLX",
    "INTC":"INTC",
    "AMD":"AMD",
    "BA":"BA",
    "JPM":"JPM",
    "DIS":"DIS",
    "V":"V",
    "NKE":"NKE"
}

def load_latest_data(stock_symbol):
    df = pd.read_csv(f"{stock_symbol}_price_sentiment.csv")
    latest = df.iloc[-1:]

    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']
    x = latest[features]

    scaler = StandardScaler()
    historical = df[features][:-1]
    scaler.fit(historical)
    x_scaled = scaler.transform(x)

    return x_scaled.reshape((1, 1, x_scaled.shape[1]))

with open("predictions.txt", "a") as f:
    f.write(f"\n--- Predictions from {datetime.now().strftime('%Y-%m-%d')} ---\n")

    for stock_symbol, model_name in stock_models.items():
            print(f"Loading model for {stock_symbol}\n")

            model = mlflow.tensorflow.load_model(model_uri=f"models:/{model_name}/1")
            x_input = load_latest_data(stock_symbol)

            pred = model.predict(x_input)[0][0]
            movement = "Up" if pred > 0.5 else "Down"

            f.write(f"{stock_symbol}: Predicted movement = {movement} ({pred:.4f})\n")
