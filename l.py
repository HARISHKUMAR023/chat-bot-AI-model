import json
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D

# Load intents.json
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocess data
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

# Prepare training data
training_sentences = []
training_labels = []

for document in documents:
    word_patterns, intent_tag = document
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    training_sentences.append(' '.join(word_patterns))
    training_labels.append(intent_tag)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
max_length = max(len(seq) for seq in sequences)
X_train = pad_sequences(sequences, maxlen=max_length, padding='post')

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(training_labels)

# Build and train the model
model = Sequential([
    Embedding(input_dim=2000, output_dim=64, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
model.save('chatbot_model.h5')

# Function to predict the class of a given text
def predict_class(text):
    tokenized_text = tokenizer.texts_to_sequences([text])
    padded_text = pad_sequences(tokenized_text, maxlen=max_length, padding='post')
    prediction = model.predict(padded_text)
    return classes[np.argmax(prediction)]

# Function to get the response for a predicted class
def get_response(intents, tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# Chatbot response function
def chatbot_response(text):
    tag = predict_class(text)
    return get_response(intents, tag)

# Example usage
print(chatbot_response("Hi"))
