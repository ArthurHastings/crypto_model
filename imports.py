import os
import datetime
from datetime import datetime, timedelta, timezone
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from dataset_concat import preprocess_text
import re
import gensim.downloader as api
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import requests
import ssl
import copy
import itertools
import mlflow
import mlflow.tensorflow
import time
from tensorflow.python.platform import build_info as tf_build_info
import pickle
import json
import yfinance as yf
from sklearn.preprocessing import StandardScaler


ssl._create_default_https_context = ssl._create_unverified_context

# Dynamically define NLTK data path
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)

# Ensure required resources are available
try:
    nltk.corpus.stopwords.words('english')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

try:
    nltk.corpus.wordnet.words()
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)
    
glove_model = api.load("glove-wiki-gigaword-100")