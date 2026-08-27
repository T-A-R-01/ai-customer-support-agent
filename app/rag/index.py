from dataclasses import dataclass

import numpy as np

from app.rag.chunker import Chunk


@dataclass
class IndexedChunk:
    """A chunk together with its embedding vector."""

    chunk: Chunk
    embedding: list[float]


class VectorIndex:
    """Simple in-memory vector index for semantic search."""

    def __init__(self) -> None:
        self.items: list[IndexedChunk] = []
        self.matrix: np.ndarray | None = None

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Add chunks and their embeddings to the index."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        self.items = [
            IndexedChunk(
                chunk=chunk,
                embedding=embedding,
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        if embeddings:
            self.matrix = np.array(embeddings, dtype=np.float32)
        else:
            self.matrix = None

    def __len__(self) -> int:
        """Return the number of indexed chunks."""

        return len(self.items)