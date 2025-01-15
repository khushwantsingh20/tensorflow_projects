# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.datasets import cifar10

# # Load CIFAR-10 dataset
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Normalize data
# X_train, X_test = X_train / 255.0, X_test / 255.0

# # Build the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # Convolutional layer
#     MaxPooling2D((2, 2)),                                            # Max pooling layer
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),                                                       # Flatten layer
#     Dense(128, activation='relu'),                                   # Fully connected layer
#     Dense(10, activation='softmax')                                  # Output layer
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# model.fit(X_train, y_train, epochs=10, batch_size=64)

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print("Test Accuracy:", accuracy)




from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1, batch_size=64)  # Reduced epochs for quick testing

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predict a sample from the test dataset
sample_index = 0  # Select the first test image
sample_image = X_test[sample_index]  # Select the image
sample_label = y_test[sample_index]  # True label of the image

# Display the image
plt.imshow(sample_image)
plt.title("True Label: " + str(sample_label[0]))
plt.show()

# Expand dimensions to make it compatible with the model input
sample_image = np.expand_dims(sample_image, axis=0)  # Shape becomes (1, 32, 32, 3)

# Predict the class
predictions = model.predict(sample_image)  # Model returns probabilities for all classes
predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability

print("Predicted Class:", predicted_class[0])
print("Prediction Probabilities:", predictions)
