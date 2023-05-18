import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

factory = StemmerFactory()
stemmer = factory.create_stemmer()


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"
                           u"\U0001F300-\U0001F5FF"
                           u"\U0001F680-\U0001F6FF"
                           u"\U0001F1E0-\U0001F1FF"
                           "]+", flags=re.UNICODE)
    
def cleansing(text):
    if isinstance(text, str):
        text = text.strip('')
        text = re.sub(r"[?|$|.|~!_:')(-+,]", '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\b[a-zA-Z]\b', '', text)
        text = re.sub('\s+',' ', text)
        text = re.sub('RT','', text)
        text = re.sub('@[^\s]+','', text)
        text = re.sub(r'https\S+', '', text)
        text = re.sub(r'#\w+\s*', '', text)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'\u2026', '', text)
        text = re.sub('/', '', text)
        text = re.sub('%', '', text)
        text = re.sub('-', '', text)
        text = re.sub("'", '', text)
        text = re.sub("#", '', text)
        text = re.sub("—", '', text)
        text = re.sub("ðŸ¤”", '', text)
        text = re.sub("ðŸ¤¦", '', text)
        text = re.sub("£", '', text)
        text = re.sub("â€“", '', text)
        text = re.sub("ðŸ¤£", '', text)
        text = re.sub("ðŸ¤—", '', text)
        text = re.sub("ðŸ¥°", '', text)
        text = re.sub("@", '', text)
        text = re.sub("ðŸ¤§", '', text)
        text = re.sub("ðŸ¥°", '', text)
        text = re.sub("ðŸ§‘â€ðŸ¦¯", '', text)
        text = re.sub("ðŸ¤²", '', text)
        text = re.sub("ðŸ¤®", '', text)
        text = re.sub("donkâ€¼ï", '', text)
        text = re.sub("ðŸ¤®", '', text)
    return text

def case_folding(text):
    if isinstance(text, str):
        text = text.lower()   
    return text     

def remove_stopwords(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
        return ' '.join(filtered_tokens)
    else:
        return text
    
@app.route('/api/pre-processing', methods=['POST'])
def pre_processing():
    json_data = request.get_json()
    data = json_data.get('data')

    remove_character = []
    text_lower_case = []
    tokens = []
    stopwords_list = []
    stemmed_tokens = []
    stemmed_text = []

    for tweet in data:
        modified_tweet = {
            'full_text': cleansing(tweet.get('full_text'))
        }
        remove_character.append(modified_tweet)
        
    for tweet in remove_character:
        modified_tweet = {
            'full_text': case_folding(tweet.get('full_text'))
        }
        text_lower_case.append(modified_tweet)
        
    for tweet in text_lower_case:
        modified_tweet = {
            'full_text': word_tokenize(tweet.get('full_text'))
        }
        tokens.append(modified_tweet)
        
    for tweet in tokens:
        modified_tweet = {
            'full_text': remove_stopwords(tweet.get('full_text'))
        }
        stopwords_list.append(modified_tweet)
        
    for tweet in stopwords_list:
        modified_tweet = {
            'full_text': [stemmer.stem(token) for token in tweet.get('full_text')]
        }
        stemmed_tokens.append(modified_tweet)
        
    for tweet in stopwords_list:
        modified_tweet = {
            'full_text': ' '.join(tweet.get('full_text'))
        }
        stemmed_text.append(modified_tweet)

    return jsonify({
        'remove_character': remove_character,
        'text_lower_case': text_lower_case,
        'tokens': tokens,
        'stopwords_list': stopwords_list,
        'stemmed_tokens': stemmed_tokens,
        'stemmed_text': stemmed_text  # Include the stemmed text in the response
    })

@app.route('/api/processing', methods=['POST'])
def processing():
    new_data = request.get_json()
    
    return jsonify(new_data)


@app.route('/', methods=['GET'])
def hello_world():
    hello = "Hello World"
    
    return hello

if __name__ == '__main__':
    app.run(debug=True)
