import ollama


print("Testing Ollama embedding model...")

response = ollama.embed(
    model="nomic-embed-text",
    input="How long do I have to return an item?",
)

embedding = response["embeddings"][0]

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")