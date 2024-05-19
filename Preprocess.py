import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()

def preprocess_data(intents):
    words = []
    classes = []
    documents = []
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ["?", "!"]]
    words = sorted(set(words))
    classes = sorted(set(classes))
    
    return words, classes, documents

with open('intents.json') as file:
    intents = json.load(file)

words, classes, documents = preprocess_data(intents)
