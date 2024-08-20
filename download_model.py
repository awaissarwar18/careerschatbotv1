from transformers import pipeline

# Download the model and tokenizer
generator = pipeline('text-generation', model='gpt2')
print("Model downloaded and cached locally.")
