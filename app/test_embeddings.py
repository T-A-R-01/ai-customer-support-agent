from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService


documents = load_knowledge_base()
chunks = chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

embedding_service = EmbeddingService()

print("\nGenerating one test embedding...")

vector = embedding_service.embed_text(
    "How long do I have to return an item?"
)

print(f"Embedding dimensions: {len(vector)}")
print(f"First 5 values: {vector[:5]}")