import mlflow
import os

mlflow_ngrok = os.getenv("MLFLOW_NGROK", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_ngrok)

model_names = [
    "crypto_sentiment_model_1",
    "AAPL",
    "AMD",
    "AMZN",
    "BA",
    "DIS",
    "GOOGL",
    "INTC",
    "JPM",
    "META",
    "MSFT",
    "NFLX",
    "NKE",
    "NVA",
    "TSLA",
    "V"
]

output_dir = os.path.join("mlruns", "models")
os.makedirs(output_dir, exist_ok=True)

for model_name in model_names:
    model_path = os.path.join(output_dir, model_name)
    print(f"Downloading {model_name} to {model_path}...")
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/latest",
        dst_path=model_path
    )
