import random
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents from the JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and lemmatize the words and classify them
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and filter unwanted characters from the words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save the words and classes as pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns if w not in ignore_letters]

    # Create a bag-of-words representation
    bag = [1 if word in word_patterns else 0 for word in words]

    # One-hot encode the output
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle the data
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into inputs and labels
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

print(f"Training data prepared: train_x shape = {train_x.shape}, train_y shape = {train_y.shape}")

# Define the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbotmodel.h5", hist)

print("Model training complete and saved as 'chatbotmodel.h5'")
