import requests
import json
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataset_concat import preprocess_text
import numpy as np

url = "https://5d75-213-233-110-28.ngrok-free.app/invocations"

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 50

sentence_example = "Bitcoin price will increase tomorrow by 5%."

sentence_example_clean = preprocess_text(sentence_example)
sentence_example_seq = tokenizer.texts_to_sequences([sentence_example_clean])
sentence_example_pad = pad_sequences(sentence_example_seq, maxlen=max_length, padding='post')

data = {
  "instances": sentence_example_pad.tolist()
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, data=json.dumps(data), headers=headers)
response_json = response.json()

predictions = response_json.get("predictions", [[]])[0]

predicted_label = np.argmax(predictions)
label_reverse_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
print(f"Predicted Sentiment: {label_reverse_mapping[predicted_label]}\n{predictions}")
