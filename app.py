import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# --- One-time NLTK data download for deployment ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
# --- End of download section ---


# --- Chatbot Core Logic ---

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents file with UTF-8 encoding
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load pre-processed data
try:
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    with open('train_x.pkl', 'rb') as f:
        train_x = pickle.load(f)
    with open('train_y.pkl', 'rb') as f:
        train_y = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py and commit the .pkl files to your GitHub repository.")
    st.stop()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence, words, classes, train_x, train_y):
    p = bag_of_words(sentence, words)
    
    max_similarity = -1
    best_match_index = -1
    
    if np.linalg.norm(p) == 0:
        return 'fallback'

    for i, train_bag in enumerate(train_x):
        norm_train_bag = np.linalg.norm(train_bag)
        if norm_train_bag == 0:
            continue
            
        similarity = np.dot(p, train_bag) / (np.linalg.norm(p) * norm_train_bag)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_index = i
            
    if max_similarity > 0.7:
        predicted_class_index = np.argmax(train_y[best_match_index])
        return classes[predicted_class_index]
    else:
        return 'fallback'

def get_response(intent_tag, intents_json):
    list_of_intents = intents_json['intents']
    result = "I'm sorry, I don't have information on that topic yet."
    for i in list_of_intents:
        if i['tag'] == intent_tag:
            result = "\n\n".join(i['responses'])
            break
    return result

# --- Streamlit Web App Interface ---

st.set_page_config(page_title="Programming Helper Bot", page_icon="🤖")

st.title("Programming Helper Bot 🤖")
st.caption("Your friendly assistant for basic Java and Python questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    startup_response = get_response('startup_message', intents)
    st.session_state.messages.append({"role": "assistant", "content": startup_response})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        intent = predict_intent(prompt, words, classes, train_x, train_y)
        response = get_response(intent, intents)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
