from transformers import pipeline

# Load the sentiment-analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze text sentiment
result = sentiment_analyzer(" it is not possible that I don't love you")
print(result)
