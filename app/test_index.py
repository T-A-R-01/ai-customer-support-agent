from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex


print("Loading knowledge base...")

documents = load_knowledge_base()
chunks = chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

print("\nGenerating embeddings...")

embedding_service = EmbeddingService()
embeddings = embedding_service.embed_chunks(chunks)

print(f"Embeddings: {len(embeddings)}")
print(f"Dimensions: {len(embeddings[0])}")

print("\nBuilding vector index...")

index = VectorIndex()

index.add(
    chunks=chunks,
    embeddings=embeddings,
)

print(f"Indexed chunks: {len(index)}")
print(f"Matrix shape: {index.matrix.shape}")