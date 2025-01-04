import nltk
import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow import keras  # Updated import

lemmatizer = WordNetLemmatizer()

# Load data
with open('intents.json') as json_file:
    intents = json.load(json_file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbotmodel.h5')  # Use keras to load model

# Function to clean up and tokenize input sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Create a bag of words representation
def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


# Predict the class of a sentence
def predict_class(sentence):
    try:
        print(f"Received sentence for prediction: {sentence}")
        bow_input = bow(sentence, words)
        print(f"Bag of Words: {bow_input}")

        prediction = model.predict(np.array([bow_input]))[0]
        print(f"Prediction Probabilities: {prediction}")

        ERROR_THRESHOLD = 0.1
        results = [(i, p) for i, p in enumerate(prediction) if p > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)

        if not results:
            print("No class prediction met the confidence threshold.")
            return None

        predicted_index = results[0][0]
        predicted_class = classes[predicted_index]
        print(f"Predicted Class: {predicted_class}")
        return predicted_class  # Return only the class tag
    except Exception as e:
        print(f"Error in predict_class: {e}")
        return None



# Get a response based on the predicted class
def get_response(tag):
    if tag is None:
        return "Sorry, I didn't understand that. Could you please rephrase?"
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I couldn't process that. Please try again."


# Main loop for chatbot interaction
print("GO! BOT IS RUNNING")

while True:
    message = input("You: ")
    if message.lower() == "exit":
        print("Goodbye!")
        break
    predictions = predict_class(message)
    response = get_response(predictions)
    print(f"Bot: {response}")
