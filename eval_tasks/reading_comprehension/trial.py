from transformers import pipeline

pipe = pipeline("text-generation", model="google/gemma-2b", trust_remote_code=True)
output = pipe("What is the capital of France?")
print(output)
