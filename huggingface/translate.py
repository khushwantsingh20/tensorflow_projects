from transformers import pipeline

# Use a specific model for translation from English to Hindi
en_hi_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
result = en_hi_translator("How old are you?")
print(result)
