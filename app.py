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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
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

def preprocess_and_split_data(texts, data):
    # Tokenize the text using a regular expression tokenizer
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(texts)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(text_counts, [tweet['label'] for tweet in data], test_size=0.2, random_state=19)

    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train):
    # Train a Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    return clf

def predict_labels(clf, X_test):
    # Make predictions on the testing data
    y_pred = clf.predict(X_test)

    return y_pred

def compute_precision(y_true, y_pred):
    # Compute precision
    precision = precision_score(y_true, y_pred, average='weighted')

    return precision

def compute_recall(y_true, y_pred):
    # Compute recall
    recall = recall_score(y_true, y_pred, average='weighted')

    return recall

def compute_accuracy(y_true, y_pred):
    # Compute accuracy
    accuracy = accuracy_score(y_true, y_pred)

    return accuracy
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
def process_data():
    json_data = request.get_json()
    data = json_data.get('data')

    # Extract tweet texts from data
    tweets = [tweet['Tweet Text'] for tweet in data]

    # Perform text preprocessing and split into training and testing sets
    X_train, X_test, y_train, y_test = preprocess_and_split_data(tweets, data)

    # Train a classifier and make predictions
    clf = train_classifier(X_train, y_train)
    y_pred = predict_labels(clf, X_test)

    # Compute precision, recall, and accuracy
    precision = compute_precision(y_test, y_pred)
    recall = compute_recall(y_test, y_pred)
    accuracy = compute_accuracy(y_test, y_pred)

    # Compute and return the confusion matrix, precision, recall, and accuracy as a JSON response
    cm = confusion_matrix(y_test, y_pred)
    response = {
        'confusion_matrix': cm.tolist(),
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'processed_data': data
    }
    
    return jsonify(response)

@app.route('/', methods=['GET'])
def hello_world():
    hello = "Hello World"
    
    return hello

if __name__ == '__main__':
    app.run(debug=True)
