import mlflow
from mlflow.tracking import MlflowClient
import os

mlflow_ngrok = os.getenv("MLFLOW_NGROK", "http://localhost:5001")
mlflow.set_tracking_uri(mlflow_ngrok)

client = MlflowClient()

model_names = [
    "crypto_sentiment_model_1",
    "AAPL", "AMD", "AMZN", "BA", "DIS", "GOOGL",
    "INTC", "JPM", "META", "MSFT", "NFLX", "NKE",
    "NVDA", "TSLA", "V"
]

output_dir = os.path.join("mlruns", "models")
os.makedirs(output_dir, exist_ok=True)

for model_name in model_names:
    try:
        print(f"\nResolving 'Production' version of model: {model_name}...")
        versions = client.get_model_version_by_alias(model_name, "production")
        run_id = versions.run_id
        source_path = versions.source

        dst_path = os.path.join(output_dir, model_name)
        print(f"Downloading artifacts from run {run_id} to {dst_path}...")

        mlflow.artifacts.download_artifacts(
            artifact_uri=source_path,
            dst_path=dst_path
        )
        print(f"✅ {model_name} downloaded to {dst_path}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")

# ZZZZZ