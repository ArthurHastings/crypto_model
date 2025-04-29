import subprocess
import mlflow

# Set the MLFLOW_TRACKING_URI environment variable for the CLI
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_tracking_uri("http://localhost:5003")

# Command to serve the model
command = [
    "mlflow", "models", "serve",
    "-m", "models:/crypto_sentiment_model_1/1",  # Model URI
    "--host", "0.0.0.0",  # Host on all interfaces
    "--port", "5002"  # Port to run the server
]

# Run the command
subprocess.run(command)

# TO INSTALL PYENV
# Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
