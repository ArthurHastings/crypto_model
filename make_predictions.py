import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf  # Needed for loading Keras models via MLflow
from datetime import datetime
import os

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
    latest = df.iloc[-1:]  # last row

    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']
    x = latest[features]

    scaler = StandardScaler()
    historical = df[features][:-1]
    scaler.fit(historical)
    x_scaled = scaler.transform(x)

    return x_scaled.reshape((1, 1, x_scaled.shape[1]))

def load_from_date(stock_symbol, date):
    df = pd.read_csv(f"{stock_symbol}_price_sentiment.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    from_date = pd.to_datetime(date)

    present_data = df[df['Date'] == from_date]
    
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Negative', 'Neutral', 'Positive']

    historical = df[df['Date'] < from_date][features]

    scaler = StandardScaler()
    scaler.fit(historical)
    x_scaled = scaler.transform(present_data[features])

    return x_scaled.reshape((x_scaled.shape[0], 1, x_scaled.shape[1])), present_data['Date'].tolist()


# pui toate datele pe care vrei sa ti le genereze de prezis pana la to_date (pui ultima data de care ai
# nevoie ex: pui 2025-05-22 daca ai nevoie de date pana 2025-05-21 inclusiv)
cont = 0
day_list = ["21", "22"]
from_date_str = f"2025-05-{day_list[cont]}"
from_date = pd.to_datetime(from_date_str)
to_date = pd.to_datetime("2025-05-22")

while from_date != to_date:
    from_date_str = f"2025-05-{day_list[cont]}"
    from_date = pd.to_datetime(from_date_str)

    with open("predictions.txt", "a") as f:
        f.write(f"\n--- Predictions from {from_date_str} ---\n")

        for stock_symbol, model_name in stock_models.items():
            print(f"Loading model for {stock_symbol}\n")
            try:
                model = mlflow.tensorflow.load_model(model_uri=f"models:/{model_name}/1")
                x_inputs, dates = load_from_date(stock_symbol, from_date_str)

                preds = model.predict(x_inputs).flatten()

                for date, pred in zip(dates, preds):
                    movement = "Up" if pred > 0.5 else "Down"
                    f.write(f"{stock_symbol}: Predicted movement = {movement} ({pred:.4f})\n")

            except Exception as e:
                print(f"Error processing {stock_symbol}: {e}")
                f.write(f"{stock_symbol}: Error - {e}\n")

    cont += 1

