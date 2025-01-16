from transformers import pipeline

# Load an image classification pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
result = classifier("cat.jpeg")
print(result)
