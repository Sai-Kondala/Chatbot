# AI-Powered Chatbot using NLTK and spaCy
# This chatbot is designed to handle both FAQs and small talk.

# Import necessary libraries
import random
import json
import pickle
import numpy as np

# NLTK for natural language processing tasks
import nltk
from nltk.stem import WordNetLemmatizer

# spaCy for more advanced NLP features
import spacy

# Load the English language model for spaCy
nlp = spacy.load('en_core_web_sm')

# Initialize the lemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# Load the intents from a JSON file
# This file contains predefined patterns and responses for the chatbot
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Initialize lists to store our data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the pair (word_list, intent['tag']) to documents
        documents.append((word_list, intent['tag']))
        # Add the tag to the classes list if it's not already there
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words, convert to lowercase, and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))

# Sort classes to maintain consistency
classes = sorted(list(set(classes)))

# Save the processed words and classes to pickle files
# This saves us from having to re-process the data every time
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare the training data
training = []
output_empty = [0] * len(classes)

# Create a bag of words for each document
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatize and convert to lowercase
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        # If the word is in the pattern, add 1 to the bag, otherwise add 0
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create the output row for the training data
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle the training data and convert to a NumPy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split the data into training and testing sets
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# This is a simplified model for demonstration.
# For a production-level chatbot, you would use a deep learning model
# like a neural network built with TensorFlow or PyTorch.
# For this example, we'll simulate the model's prediction process.

def clean_up_sentence(sentence):
    """Tokenizes and lemmatizes the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """Creates a bag of words from the input sentence."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_intent(sentence):
    """Predicts the intent of the sentence."""
    bow = bag_of_words(sentence)
    # This is a simplified prediction. A real model would have a predict function.
    # We find the training data entry with the most similar bag of words.
    max_similarity = -1
    best_match_index = -1
    
    for i, train_bag in enumerate(train_x):
        # Using cosine similarity for comparison
        similarity = np.dot(bow, train_bag) / (np.linalg.norm(bow) * np.linalg.norm(train_bag))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_index = i
            
    # If the similarity is high enough, return the predicted intent
    if max_similarity > 0.7: # Confidence threshold
        predicted_class_index = np.argmax(train_y[best_match_index])
        return classes[predicted_class_index]
    return None

def get_response(intent_tag, intents_json):
    """Fetches a random response for the given intent."""
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == intent_tag:
            result = random.choice(i['responses'])
            break
    return result

# Main loop for the chatbot
print("Chatbot is running! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        break
    
    # Predict the intent of the user's message
    predicted_intent = predict_intent(message)
    
    if predicted_intent:
        # If an intent is found, get a response
        response = get_response(predicted_intent, intents)
    else:
        # If no intent is matched, provide a default response
        response = "I'm not sure how to respond to that. Can you ask me something else?"
        
    print(f"Bot: {response}")

