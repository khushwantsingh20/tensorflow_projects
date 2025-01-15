from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode input text
input_text = "give me the Hello world code in python"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate a response
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
