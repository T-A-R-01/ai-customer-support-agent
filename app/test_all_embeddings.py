import time

from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService


documents = load_knowledge_base()
chunks = chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

embedding_service = EmbeddingService()

print("\nGenerating embeddings for all chunks...")

start = time.time()

embeddings = embedding_service.embed_chunks(chunks)

elapsed = time.time() - start

print(f"Generated embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"Time taken: {elapsed:.2f} seconds")