from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever
from app.rag.safety import EvidenceChecker
from app.rag.generator import Generator


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


print("Creating evidence checker...")

checker = EvidenceChecker()


print("Creating generator...")

generator = Generator()


queries = [
    "How long do I have to return an item?",
    "What is the weather on Mars tomorrow?",
]


for query in queries:

    print("\n========================================")
    print(f"QUESTION: {query}")
    print("========================================")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print("\nRetrieved context:")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"semantic={result.semantic_score:.4f} | "
            f"ranking={result.score:.4f} | "
            f"{result.chunk.source} | "
            f"{result.chunk.heading}"
        )

    evidence = checker.check(results)

    print("\nEvidence check:")
    print(f"Sufficient: {evidence.sufficient}")
    print(f"Conflict: {evidence.conflict}")
    print(f"Reason: {evidence.reason}")

    # ---------------------------------------------------------
    # Genuine conflict
    # ---------------------------------------------------------

    if evidence.conflict:

        print("\n========== CONFLICT ==========\n")

        print(
            "The available authoritative sources contain "
            "conflicting information. Human confirmation "
            "is recommended."
        )

        continue

    # ---------------------------------------------------------
    # Insufficient / irrelevant evidence
    # ---------------------------------------------------------

    if not evidence.sufficient:

        print("\n========== ABSTAIN ==========\n")

        print(
            "I don't have enough reliable information "
            "in the knowledge base to answer this question."
        )

        continue

    # ---------------------------------------------------------
    # Safe to generate an answer
    # ---------------------------------------------------------

    print("\nGenerating final answer...")

    answer = generator.generate(
        query=query,
        results=results,
    )

    print("\n========== FINAL ANSWER ==========\n")
    print(answer)