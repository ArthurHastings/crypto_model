FROM python:3.11.0

ENV MLFLOW_TRACKING_URI="http://localhost:5003"

RUN pip install --upgrade pip==23.2.1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./mlruns/models /app/models

ENV MODEL_NAME=crypto

EXPOSE 5001

CMD ["sh", "-c", "mlflow models serve --host 0.0.0.0 --port 5001 --model-uri file:///app/models/$MODEL_NAME --no-conda"]

