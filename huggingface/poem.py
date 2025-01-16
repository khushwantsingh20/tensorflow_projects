# text_generator.py

from transformers import pipeline

# Initialize the text generation pipeline with GPT-2 (you can also use GPT-3 if available)
generator = pipeline('text-generation', model='gpt2')

def generate_text(prompt, max_length=200):
    """
    Generates text based on the input prompt using GPT-2.
    """
    print("\nGenerating text...\n")
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]['generated_text']

def main():
    print("Welcome to the AI Text Generator!")
    print("This tool will generate text based on the prompt you provide.")
    print("Type 'exit' to quit the program.")
    
    while True:
        # Get user input for prompt
        user_input = input("\nEnter a prompt: ")
        
        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the AI Text Generator. Goodbye!")
            break
        
        # Generate and display text
        generated_text = generate_text(user_input)
        print("\nGenerated Text:\n")
        print(generated_text)

if __name__ == "__main__":
    main()
