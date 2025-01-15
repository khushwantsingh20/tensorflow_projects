from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDB dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

# Build the RNN model
model = Sequential([
    Embedding(10000, 128, input_length=100),  # Embedding layer
    SimpleRNN(64, activation='relu'),         # RNN layer
    Dense(1, activation='sigmoid')            # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Example of Prediction
def predict_sentiment(review):
    # Get the word index from IMDB dataset
    word_index = imdb.get_word_index()

    # Tokenize and convert the review to indices
    review_words = review.split()
    review_indices = [word_index.get(word.lower(), 0) for word in review_words]  # Default to 0 if word is not found

    # Ensure all indices are within the range [0, 9999] (valid indices for the embedding layer)
    review_indices = [min(index, 9999) for index in review_indices]

    # Pad the sequence to ensure it matches the input length (100)
    review_padded = pad_sequences([review_indices], maxlen=100)

    # Predict sentiment
    prediction = model.predict(review_padded)
    
    # Return the sentiment
    if prediction > 0.5:
        return "Positive Review"
    else:
        return "Negative Review"

# Example usage:
sample_review = "DC is good but marvel is the best"
print(predict_sentiment(sample_review))
