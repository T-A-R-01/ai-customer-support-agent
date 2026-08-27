from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever


documents = load_knowledge_base()
chunks = chunk_documents(documents)

embedding_service = EmbeddingService()
embeddings = embedding_service.embed_chunks(chunks)

index = VectorIndex()
index.add(
    chunks=chunks,
    embeddings=embeddings,
)

retriever = Retriever(
    index=index,
    embedding_service=embedding_service,
)

query = "What is the current return policy?"

results = retriever.search(
    query=query,
    top_k=10,
)

print(f"\nQuery: {query}\n")

for rank, result in enumerate(results, start=1):
    metadata = result.chunk.metadata

    print(f"--- Result {rank} ---")
    print(f"Source: {result.chunk.source}")
    print(f"Document ID: {metadata.get('document_id')}")
    print(f"Status: {metadata.get('status')}")
    print(f"Authority: {metadata.get('policy_authority')}")
    print(f"Score: {result.score:.4f}")
    print()