import numpy as np

from app.rag.chunker import Chunk
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex


class RetrievalResult:
    """A retrieved chunk together with its retrieval scores."""

    def __init__(
        self,
        chunk: Chunk,
        score: float,
        semantic_score: float,
    ) -> None:
        self.chunk = chunk
        self.score = score
        self.semantic_score = semantic_score


class Retriever:
    """Retrieve authoritative, relevant knowledge-base chunks."""

    def __init__(
        self,
        index: VectorIndex,
        embedding_service: EmbeddingService,
    ) -> None:
        self.index = index
        self.embedding_service = embedding_service

    def _is_customer_usable(
        self,
        chunk: Chunk,
    ) -> bool:
        """Return whether a chunk can be used for a customer answer."""

        metadata = chunk.metadata

        if metadata.get("customer_answering") is False:
            return False

        if metadata.get("policy_authority") == "none":
            return False

        if metadata.get("status") == "superseded":
            return False

        return True

    def _authority_bonus(
        self,
        chunk: Chunk,
    ) -> float:
        """Give authoritative active customer content a ranking bonus."""

        metadata = chunk.metadata

        bonus = 0.0

        if metadata.get("policy_authority") == "official":
            bonus += 0.10

        if metadata.get("status") == "active":
            bonus += 0.10

        if metadata.get("audience") == "customer":
            bonus += 0.05

        return bonus

    def _lexical_bonus(
        self,
        query: str,
        chunk: Chunk,
    ) -> float:
        """Boost chunks that share meaningful words with the query."""

        query_words = {
            word.lower().strip(".,!?;:()[]{}\"'")
            for word in query.split()
            if len(word.strip(".,!?;:()[]{}\"'")) >= 4
        }

        text = (
            f"{chunk.heading} "
            f"{chunk.text}"
        ).lower()

        if not query_words:
            return 0.0

        matches = sum(
            1
            for word in query_words
            if word in text
        )

        return min(matches * 0.015, 0.08)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return the most relevant customer-usable chunks."""

        if self.index.matrix is None:
            return []

        if not self.index.items:
            return []

        # ---------------------------------------------------------
        # 1. Convert the user's question into an embedding.
        # ---------------------------------------------------------

        query_embedding = self.embedding_service.embed_text(
            query
        )

        query_vector = np.array(
            query_embedding,
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # 2. Calculate cosine similarity against every chunk.
        # ---------------------------------------------------------

        scores = self.index.matrix @ query_vector

        matrix_norms = np.linalg.norm(
            self.index.matrix,
            axis=1,
        )

        query_norm = np.linalg.norm(
            query_vector
        )

        similarities = scores / (
            matrix_norms * query_norm + 1e-10
        )

        # ---------------------------------------------------------
        # 3. Build candidates.
        #
        # Keep semantic similarity separate from the final
        # ranking score.
        # ---------------------------------------------------------

        candidates = []

        for index, similarity in enumerate(
            similarities
        ):
            chunk = self.index.items[index].chunk

            # Do not use superseded/internal/non-authoritative
            # content for customer answers.
            if not self._is_customer_usable(chunk):
                continue

            semantic_score = float(similarity)

            authority_score = self._authority_bonus(
                chunk
            )

            lexical_score = self._lexical_bonus(
                query,
                chunk,
            )

            ranking_score = (
                semantic_score
                + authority_score
                + lexical_score
            )

            candidates.append(
                (
                    ranking_score,
                    semantic_score,
                    chunk,
                )
            )

        # ---------------------------------------------------------
        # 4. Rank candidates using the combined score.
        # ---------------------------------------------------------

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # ---------------------------------------------------------
        # 5. Limit duplicate chunks from the same document.
        # ---------------------------------------------------------

        results = []

        source_counts: dict[str, int] = {}

        max_chunks_per_source = 2

        for (
            ranking_score,
            semantic_score,
            chunk,
        ) in candidates:

            source = chunk.source

            current_count = source_counts.get(
                source,
                0,
            )

            if current_count >= max_chunks_per_source:
                continue

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=ranking_score,
                    semantic_score=semantic_score,
                )
            )

            source_counts[source] = (
                current_count + 1
            )

            if len(results) >= top_k:
                break

        return results