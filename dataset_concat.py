import nltk

required_resources = ['punkt', 'stopwords', 'wordnet']
for resource in required_resources:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset
import re

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9$%. ]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

if __name__ == "__main__":

    df_og4 = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
    df_og4 = pd.DataFrame(df_og4["train"])
    df_og5 = load_dataset("neeeeellllll/Financial_Sentiment_Analyis_dataset", trust_remote_code=True)
    df_og5 = pd.DataFrame(df_og5["train"])

    df_og4.rename(columns={'sentence': 'Sentence', 'label':'Sentiment'}, inplace=True)

    label_mapping = {0: "negative", 1: 'neutral', 2:'positive'}
    df_og4['Sentiment'] = df_og4['Sentiment'].map(label_mapping)
    df_og5['Sentiment'] = df_og5['Sentiment'].map(label_mapping)

    df_og4 = df_og4.dropna(subset=['Sentiment'])
    df_og5 = df_og4.dropna(subset=['Sentiment'])

    # df_og = pd.concat([df_og, df_og2, df_og3, df_og4], ignore_index=True)
    df_og = pd.concat([df_og4, df_og5], ignore_index=True)

    df_og.reset_index(drop=True, inplace=True)

    df_og['cleaned_text'] = df_og['Sentence'].apply(preprocess_text)
    df_og = df_og.drop("Sentence", axis=1)

    def remove_spaces(text: str):
        result = []
        i = 0
        while i < len(text):
            if text[i] == "$" and i + 1 < len(text) and text[i + 1] == " ":
                result.append(text[i])
                i += 2
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    df_og.cleaned_text = df_og.cleaned_text.apply(remove_spaces)


    df_og.to_csv(r"data_final.csv", index=False)