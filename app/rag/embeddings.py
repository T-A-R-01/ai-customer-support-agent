import ollama

from app.rag.chunker import Chunk


EMBEDDING_MODEL = "nomic-embed-text"


class EmbeddingService:
    """Generate local embeddings using Ollama."""

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for one piece of text."""

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=text,
        )

        return response["embeddings"][0]

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """Generate embeddings for all chunks."""

        texts = [chunk.text for chunk in chunks]

        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        return response["embeddings"]