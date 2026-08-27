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

queries = [
    "What is the current return policy?",
    "How long do I have to return an item?",
    "What happens if I receive a damaged product?",
    "How much does international shipping cost?",
    "What is the weather on Mars tomorrow?",
    "Who won the football match yesterday?",
    "Tell me a joke about programming.",
]

for query in queries:

    print("\n========================================")
    print(f"QUESTION: {query}")
    print("========================================")

    results = retriever.search(
        query=query,
        top_k=3,
    )

    for rank, result in enumerate(results, start=1):

        print(
            f"{rank}. "
            f"semantic={result.semantic_score:.4f} | "
            f"ranking={result.score:.4f} | "
            f"{result.chunk.source} | "
            f"{result.chunk.heading}"
        )