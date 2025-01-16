# text_classifier.py

from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define candidate labels for classification
candidate_labels = ["Technology", "Sports", "Politics", "Entertainment"]

def classify_text(text):
    """
    Classifies the input text into one of the candidate categories
    """
    result = classifier(text, candidate_labels)
    predicted_label = result['labels'][0]  # Get the top predicted label
    score = result['scores'][0]  # Confidence score of the prediction
    print(f"Predicted Category: {predicted_label} with a confidence of {score:.2f}")

def main():
    print("Welcome to the Text Classification Tool!")
    print("Classify your text into the following categories: Technology, Sports, Politics, Entertainment")
    print("Type 'exit' to quit the program.")
    
    while True:
        # Get user input
        user_input = input("\nEnter your text: ")
        
        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the Text Classification Tool. Goodbye!")
            break
        
        # Classify the entered text
        classify_text(user_input)

if __name__ == "__main__":
    main()
