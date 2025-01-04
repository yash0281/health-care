# Flask and NLP imports (unchanged)
import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__)

# Load model, intents, words, and classes
model = load_model('chatbotmodel.h5')
with open('intents.json', 'r') as file:
    intents = json.load(file)
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Helper Functions
def clean_up_sentence(sentence):
    """Tokenize and lemmatize input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    """Convert sentence to bag-of-words format."""
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

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

def get_response(intent_tag):
    """Fetch a response based on the predicted tag."""
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            response = random.choice(intent['responses'])
            print(f"Selected Response: {response}")
            return response
    print(f"No matching response found for tag: {intent_tag}")
    return "I'm sorry, I couldn't process that. Please try again."
import openai

# Set OpenAI API key
import openai
import os

# Retrieve API Key from Environment
openai.api_key = "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"

def get_openai_response(user_message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return response['choices'][0]['message']['content'].strip()



# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")  # Ensure templates/index.html exists

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    predicted_tag = predict_class(user_input)
    
    if predicted_tag:
        bot_response = get_response(predicted_tag)
    else:
        # Use OpenAI for general queries
        bot_response = get_openai_response(user_input)

    return jsonify({"response": bot_response})



if __name__ == "__main__":
    app.run(debug=True)
