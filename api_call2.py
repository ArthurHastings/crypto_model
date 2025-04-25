import requests
import json

# Replace with your ngrok URL
url = "https://5d75-213-233-110-28.ngrok-free.app/invocations"

# Prepare your input data in the correct format
data = {
    "columns": ["Close", "Open", "High", "Low", "Volume", "Negative", "Neutral", "Positive"],
    "data": [
        [150.1, 148.3, 152.2, 149.7, 1000000, 0.1, 0.2, 0.7]
    ]
}

# MLflow expects the input to be inside an 'instances' field for this scoring protocol
payload = {
    "instances": data["data"]
}

# Send the request
response = requests.post(url, json=payload)

# Check the response
print(response.status_code)
print(response.json())
