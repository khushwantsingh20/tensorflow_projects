# import tensorflow as tf
# print('TensorFlow version - ',tf.__version__)
# # Check if GPU is available
# gpu_available = tf.config.list_physical_devices('GPU')

# if gpu_available:
#     print("TensorFlow is installed as GPU version.")
# else:
#     print("TensorFlow is installed as CPU version.")


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Prepare data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize the data

# Build the model
model = Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(X_test, y_test)
