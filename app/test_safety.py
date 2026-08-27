from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever
from app.rag.safety import EvidenceChecker


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

checker = EvidenceChecker()

queries = [
    "What is the current return policy?",
    "What is the weather on Mars tomorrow?",
]

for query in queries:

    print("\n================================")
    print(f"QUESTION: {query}")
    print("================================")

    results = retriever.search(
        query=query,
        top_k=5,
    )

    check = checker.check(results)

    print(f"Sufficient: {check.sufficient}")
    print(f"Conflict: {check.conflict}")
    print(f"Reason: {check.reason}")