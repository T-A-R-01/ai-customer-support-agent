from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever


print("Loading knowledge base...")

documents = load_knowledge_base()
chunks = chunk_documents(documents)

print(f"Documents: {len(documents)}")
print(f"Chunks: {len(chunks)}")

print("\nGenerating embeddings...")

embedding_service = EmbeddingService()
embeddings = embedding_service.embed_chunks(chunks)

print("Building vector index...")

index = VectorIndex()
index.add(
    chunks=chunks,
    embeddings=embeddings,
)

print("Creating retriever...")

retriever = Retriever(
    index=index,
    embedding_service=embedding_service,
)

query = "How long do I have to return an item?"

print(f"\nQuery: {query}")
print("\nTop results:\n")

results = retriever.search(
    query=query,
    top_k=5,
)

for rank, result in enumerate(results, start=1):
    print(f"--- Result {rank} ---")
    print(f"Score: {result.score:.4f}")
    print(f"Source: {result.chunk.source}")
    print(f"Heading: {result.chunk.heading}")
    print(f"Chunk ID: {result.chunk.chunk_id}")
    print(f"Text: {result.chunk.text[:500]}")
    print()