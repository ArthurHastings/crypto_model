import subprocess
import mlflow

mlflow.set_tracking_uri("http://localhost:5003")

command = [
    "mlflow", "models", "serve",
    "-m", "models:/crypto_sentiment_model_1/1",
    "--host", "0.0.0.0",
    "--port", "5002",
    "--env-manager", "local"
]

subprocess.run(command)


# TO INSTALL PYENV
# Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
