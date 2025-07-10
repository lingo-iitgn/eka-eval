from transformers import pipeline

# Load the text-generation pipeline with Meta-Llama-3-8B
generator = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B", trust_remote_code=True)

# Provide a simple prompt
prompt = "In a future world ruled by AI,"

# Generate text
output = generator(prompt, max_new_tokens=50, do_sample=False)

# Print the result
print("Generated text:\n", output[0]['generated_text'])
