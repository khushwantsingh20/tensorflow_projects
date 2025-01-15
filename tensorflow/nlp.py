# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Embedding, Dense
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Sample text data
# text = "Deep learning is a subset of machine learning, which is a subset of artificial intelligence."

# # Tokenize text
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts([text])
# sequence = tokenizer.texts_to_sequences([text])[0]

# # Prepare data for LSTM
# X, y = [], []
# for i in range(1, len(sequence)):
#     X.append(sequence[:i])
#     y.append(sequence[i])
# X = pad_sequences(X)
# y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# # Build the LSTM model
# model = Sequential([
#     Embedding(len(tokenizer.word_index) + 1, 10, input_length=X.shape[1]),
#     LSTM(50, activation='relu'),
#     Dense(len(tokenizer.word_index) + 1, activation='softmax')
# ])

# # Compile and train the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X, y, epochs=500, verbose=1)

# # Prediction function
# def predict_next_word(text_input):
#     # Tokenize and convert the input text to sequence
#     sequence = tokenizer.texts_to_sequences([text_input])[0]
    
#     # Pad the sequence to ensure it matches the input length of the model
#     sequence_padded = pad_sequences([sequence], maxlen=X.shape[1])

#     # Predict the next word
#     pred = model.predict(sequence_padded, verbose=0)
    
#     # Get the index of the predicted word
#     pred_index = pred.argmax(axis=-1)[0]
    
#     # Map the predicted index back to a word
#     pred_word = tokenizer.index_word.get(pred_index, "<UNK>")
    
#     return pred_word

# # Example usage:
# input_text = "Deep learning is"
# predicted_word = predict_next_word(input_text)
# print(f"Next word prediction: {predicted_word}")






# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing.text import Tokenizer

# # Load the IMDB dataset
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)  # Limit to top 10,000 most frequent words

# # Tokenizer setup for the IMDB dataset
# tokenizer = Tokenizer(num_words=10000)
# X_train = tokenizer.sequences_to_texts(X_train)
# X_test = tokenizer.sequences_to_texts(X_test)

# # Tokenizing the text
# tokenizer.fit_on_texts(X_train)

# # Prepare sequences for the model
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# # Pad sequences to ensure they have the same length
# X_train_pad = pad_sequences(X_train_seq, maxlen=100)
# X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# # Build the LSTM model
# model = Sequential([
#     Embedding(input_dim=10000, output_dim=100, input_length=100),
#     LSTM(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train_pad, y_train, epochs=5, batch_size=64)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test_pad, y_test)
# print("Test Accuracy:", accuracy)

# # Next word prediction using the trained model
# def predict_next_word(input_text):
#     # Tokenize and pad the input text
#     input_seq = tokenizer.texts_to_sequences([input_text])
#     input_pad = pad_sequences(input_seq, maxlen=100)

#     # Predict the next word
#     prediction = model.predict(input_pad)

#     # Decode the predicted word index to the word
#     predicted_index = prediction.argmax(axis=-1)[0]
#     reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
#     predicted_word = reverse_word_index.get(predicted_index, '<UNK>')

#     return predicted_word

# # Example usage
# input_text = "The movie was"
# predicted_word = predict_next_word(input_text)
# print(f"Predicted next word: {predicted_word}")



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample text data (You can replace this with your dataset)
text = "Deep learning is a subset of machine learning, which is a subset of artificial intelligence."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])  # Fit tokenizer on the sample text
sequence = tokenizer.texts_to_sequences([text])[0]

# Prepare data for LSTM (sequence data preparation)
X, y = [], []
for i in range(1, len(sequence)):
    X.append(sequence[:i])
    y.append(sequence[i])

X = pad_sequences(X)  # Padding sequences for uniform length
y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Build the LSTM model
model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 10, input_length=X.shape[1]),  # Embed the sequences
    LSTM(50, activation='relu'),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=1)

# Save the model in different formats
model.save("my_model.h5")  # Save in HDF5 format
model.save("my_model.keras")  # Save in Keras native format
model.save("saved_model/my_model")  # Save in TensorFlow SavedModel format

# Prediction: Predict the next word based on the input sequence
def predict_next_word(sequence_input):
    sequence_input = tokenizer.texts_to_sequences([sequence_input])[0]
    sequence_input = pad_sequences([sequence_input], maxlen=X.shape[1], padding='pre')
    predicted = model.predict(sequence_input)
    predicted_word_index = predicted.argmax(axis=-1)[0]
    predicted_word = tokenizer.index_word.get(predicted_word_index, "<UNK>")
    return predicted_word

# Example prediction
input_sequence = "Deep learning is"
predicted_word = predict_next_word(input_sequence)
print(f"Predicted next word: {predicted_word}")
