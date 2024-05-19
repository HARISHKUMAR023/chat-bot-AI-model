from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load the trained model
model = load_model('chatbot_model.h5')

# Load intents file
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Prepare the tokenizer and the maximum sequence length
tokenizer = Tokenizer(num_words=2000)
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

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!', '.', ',']]
words = sorted(set(words))
classes = sorted(set(classes))

training_sentences = [' '.join([lemmatizer.lemmatize(w.lower()) for w in word_list]) for word_list, intent in documents]
tokenizer.fit_on_texts(training_sentences)
max_length = max(len(word_tokenize(sentence)) for sentence in training_sentences)

def preprocess_input(text):
    word_list = word_tokenize(text)
    word_list = [lemmatizer.lemmatize(w.lower()) for w in word_list]
    sequences = tokenizer.texts_to_sequences([' '.join(word_list)])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def predict_class(text):
    padded_sequences = preprocess_input(text)
    prediction = model.predict(padded_sequences)
    return classes[np.argmax(prediction)]

def get_response(intents, tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    tag = predict_class(user_message)
    response = get_response(intents, tag)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
