import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image, ImageOps
import requests
from io import BytesIO

# Function to load and preprocess an image
def load_and_preprocess_image(image_path_or_url):
    try:
        # Load image from URL or file path
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path_or_url).convert('RGB')
        
        # Resize image with padding to maintain aspect ratio
        image = ImageOps.fit(image, (224, 224), Image.LANCZOS)

        # Convert the image to a numpy array
        image_array = np.array(image)
        
        # Add a batch dimension and preprocess the image
        image_batch = np.expand_dims(image_array, axis=0)
        return preprocess_input(image_batch)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Function to predict using a pre-trained model
def predict_with_model(model, image):
    predictions = model.predict(image)
    decoded_predictions = decode_predictions(predictions, top=3)  # Top-3 predictions
    return decoded_predictions

# Function to predict with augmented images
def predict_with_augmentation(model, image):
    datagen = ImageDataGenerator(rotation_range=30, zoom_range=0.2, brightness_range=(0.8, 1.2))
    augmented_predictions = []
    
    for _ in range(5):  # Augment and predict 5 times
        for batch in datagen.flow(image, batch_size=1):
            pred = model.predict(batch)
            augmented_predictions.append(pred)
            break  # Process one batch only
    
    # Average the predictions
    averaged_predictions = np.mean(augmented_predictions, axis=0)
    decoded_predictions = decode_predictions(averaged_predictions, top=3)
    return decoded_predictions

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Path or URL to the image
image_path_or_url = "/home/khushwant/Downloads/download (2).jpeg"  # Replace with your image path or URL

# Preprocess the image
image_preprocessed = load_and_preprocess_image(image_path_or_url)

if image_preprocessed is not None:
    # Standard predictions
    print("Standard Predictions:")
    standard_predictions = predict_with_model(model, image_preprocessed)
    for i, (imagenet_id, label, score) in enumerate(standard_predictions[0]):
        print(f"{i + 1}: {label} (confidence: {score:.2f})")
    
    # Predictions with augmentation
    print("\nPredictions with Augmentation:")
    augmented_predictions = predict_with_augmentation(model, image_preprocessed)
    for i, (imagenet_id, label, score) in enumerate(augmented_predictions[0]):
        print(f"{i + 1}: {label} (confidence: {score:.2f})")
else:
    print("Failed to process the image.")
