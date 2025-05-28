import subprocess
import mlflow
import time
# are probleme cu poetry run python deploy_model.py, faci cu python depoloy_model.py
mlflow.set_tracking_uri("http://localhost:5003")

command = [
    "mlflow", "models", "serve",
    "-m", "models:/crypto_sentiment_model_1/1",
    "--host", "0.0.0.0",
    "--port", "5002",
    "--env-manager", "local"
]

subprocess.Popen(command)

time.sleep(10)


# TO INSTALL PYENV
# Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
