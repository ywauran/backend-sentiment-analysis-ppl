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
from nltk.sentiment.util import mark_negation   

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

factory = StemmerFactory()
stemmer = factory.create_stemmer()

model_dataset = joblib.load('./models/model_dataset.pkl')
vectorizer_dataset = joblib.load('./vectorizer/vectorizer_dataset.pkl')

emoji_pattern = re.compile("["
                u"\U0001F000-\U0001F9EF"  # Miscellaneous Symbols and Pictographs
                u"\U0001F300-\U0001F5FF"  # Symbols and Pictographs Extended-A
                u"\U0001F600-\U0001F64F"  # Emoticons (Emoji)
                u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                u"\U0001F700-\U0001F77F"  # Alchemical Symbols
                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-B
                u"\U00002702-\U000027B0"  # Dingbats
                u"\U0001F018-\U0001F270"  # Miscellaneous Symbols
                u"\u200d"  # Zero Width Joiner
                u"\u200c"  # Zero Width Non-Joiner
                "\U0001F602"
                "]+", flags=re.UNICODE)

def remove_emojis(text):
    if isinstance(text, str):
        return emoji_pattern.sub(r'', text)
    else:
        return text


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
    
    # Load the slang dictionary
slang_dict = pd.read_csv("slang/slanglist.csv", index_col=0)["real_word"].to_dict()

# Define a function to replace slang words with their real counterparts
def replace_slang(text):
    pattern = re.compile(r'\b(' + '|'.join(slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

# Define a function to preprocess the text data
def slang_remove(text):
    # Replace slang words with their real counterparts
    text = replace_slang(text)
    # Tokenize the text
    text = nltk.word_tokenize(text.lower())
    # Apply mark negation

    # Join the tokens back into a string
    text = " ".join(text)
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

@app.route('/api/preprocessing', methods=['POST'])
def pre_processing():
    json_data = request.get_json()
    data = json_data.get('data')
    
    remove_emoji = []
    remove_character = []
    text_lower_case = []
    tokens = []
    stopwords_list = []
    stemmed_tokens = []
    stemmed_text = []
    slang_removed_text = []
    final_process = []
    
    for tweet in data:
        modified_tweet = {
            'removeemoji_text': remove_emojis(tweet.get('full_text')),
            'original_emoji' : tweet.get('full_text')
        }
        remove_emoji.append(modified_tweet)
    
    for tweet in remove_emoji:
        modified_tweet = {
            'cleansing_text': cleansing(tweet.get('removeemoji_text')),
            'original_cleansing' : tweet.get('removeemoji_text')
        }
        remove_character.append(modified_tweet)
        
    for tweet in remove_character:
        modified_tweet = {
            'casefolding_text': case_folding(tweet.get('cleansing_text')),
            'original_casefolding' :tweet.get('cleansing_text')
        }   
        text_lower_case.append(modified_tweet)
        
    for tweet in text_lower_case:
        modified_tweet = {
            'tokenize_text': word_tokenize(tweet.get('casefolding_text')),
            'original_tokenize' : tweet.get('casefolding_text')
        }
        tokens.append(modified_tweet)
        
    for tweet in tokens:
        modified_tweet = {
            'stopwords_text': remove_stopwords(tweet.get('tokenize_text')),
            'original_stopwords' :tweet.get('tokenize_text')
        }
        stopwords_list.append(modified_tweet)
        
    for tweet in stopwords_list:
        modified_tweet = {
            'stemming_text': [stemmer.stem(token) for token in tweet.get('stopwords_text')],
            'original_stemming' : tweet.get('stopwords_text')
        }
        stemmed_tokens.append(modified_tweet)
        
    for tweet in stemmed_tokens:
        modified_tweet = {
            'join_text': ' '.join(tweet.get('stemming_text')),
            'original_jointext' : tweet.get('stemming_text')
        }
        stemmed_text.append(modified_tweet)
        
    for tweet in stemmed_text:
        modified_tweet = {
            'slangremove_andfinaltext' : slang_remove(tweet.get('join_text')),
            'original_slangandfinaltext' : tweet.get('join_text')
        }
        slang_removed_text.append(modified_tweet)
        
    for emoji_tweet, slang_tweet in zip(remove_emoji, slang_removed_text):
        modified_tweet = {
            'original_emoji': emoji_tweet.get('original_emoji'),
            'slangremove_andfinaltext': slang_tweet.get('slangremove_andfinaltext')
        }
        final_process.append(modified_tweet)
    
    return jsonify({
        'remove_character': remove_character,
        'text_lower_case': text_lower_case,
        'tokens': tokens,
        'stopwords_list': stopwords_list,
        'stemmed_tokens': stemmed_tokens,
        'stemmed_text': stemmed_text
    })

@app.route('/api/processing', methods=['POST'])
def process_data():
    json_data = request.get_json()
    data = json_data.get('data')

    # Extract tweet texts from data
    tweets = [tweet['Tweet Text'] for tweet in data]
    # Apply slang removal to the tweet texts
    tweets = [slang_remove(tweet) for tweet in tweets]

    # Perform text preprocessing and split into training and testing sets
    X_train, X_test, y_train, y_test = preprocess_and_split_data(tweets, data)

    # Train a classifier and make predictions
    clf = train_classifier(X_train, y_train)
    y_pred = predict_labels(clf, X_test)
    # Assign predicted labels to data
    for i, item in enumerate(data):
        item['predicted_label'] = y_pred[i] if i < len(y_pred) else None

        
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

@app.route('/prediction', methods=['POST'])
def predict_dataset():
    try:
        # Extract the 'data' field from the JSON payload as an array
        texts = request.json['data']

        # Perform text classification for each text in the array
        predictions = []
        true_labels = []
        for item in texts:
            text = item['join_text'].lower()  # Extract the text from the 'join_text' field

            text_counts = vectorizer_dataset.transform([text])
            predicted_label = model_dataset.predict(text_counts)[0]
            predicted_label = int(predicted_label)
            predictions.append({'sentiment': predicted_label, 'text': text})
            true_labels.append(0)  # Assuming all ground truth labels are 0 for this example
        
        # Calculate evaluation metrics
        predicted_labels = [pred['sentiment'] for pred in predictions]
        confusion = confusion_matrix(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)

        # Create the response dictionary including predictions and evaluation metrics
        response = {
            'predictions': predictions,
            'confusion_matrix': confusion.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        # Return the response dictionary as JSON response
        return jsonify(response)
    
    except Exception as e:
        # Return an error message if an exception occurs
        return jsonify({'error': str(e)})
    
@app.route('/', methods=['GET'])
def hello_world():
    hello = "Hello World"
    
    return hello

if __name__ == '__main__':
    app.run(debug=True)
