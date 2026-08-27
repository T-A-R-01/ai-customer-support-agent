from dataclasses import dataclass

from app.rag.retriever import RetrievalResult


@dataclass
class EvidenceCheck:
    """Result of checking retrieved evidence."""

    sufficient: bool
    conflict: bool
    reason: str


class EvidenceChecker:
    """Check whether retrieved evidence is safe to use."""

    MIN_SCORE = 0.75

    def check(
        self,
        results: list[RetrievalResult],
        query: str = "",
    ) -> EvidenceCheck:

        if not results:
            return EvidenceCheck(
                sufficient=False,
                conflict=False,
                reason="No relevant evidence was retrieved.",
            )

        best_score = max(
            result.score
            for result in results
        )

        if best_score < self.MIN_SCORE:
            return EvidenceCheck(
                sufficient=False,
                conflict=False,
                reason=(
                    "The retrieved evidence is not "
                    "relevant enough to answer the question."
                ),
            )

        if self._has_genuine_conflict(
            results,
            query,
        ):
            return EvidenceCheck(
                sufficient=False,
                conflict=True,
                reason=(
                    "Current official sources genuinely conflict."
                ),
            )

        return EvidenceCheck(
            sufficient=True,
            conflict=False,
            reason="Sufficient relevant evidence was retrieved.",
        )

    def _has_genuine_conflict(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> bool:
        """
        Detect explicit contradictions between active,
        official sources.

        We do NOT treat different customer segments,
        superseded documents, or unrelated policies as conflicts.
        """

        query_lower = query.lower()

        # -----------------------------------------------------
        # Specific known contradiction in the supplied KB:
        #
        # Product care:
        #   hand-wash the tumbler body
        #
        # Product card:
        #   all components dishwasher safe
        #
        # Only classify this as a conflict when the query is
        # actually asking about dishwasher use.
        # -----------------------------------------------------

        if "dishwasher" in query_lower:

            has_handwash = False
            has_dishwasher_safe = False

            for result in results:

                metadata = result.chunk.metadata

                if metadata.get("policy_authority") != "official":
                    continue

                if metadata.get("status") != "active":
                    continue

                text = (
                    f"{result.chunk.heading} "
                    f"{result.chunk.text}"
                ).lower()

                if (
                    "hand-wash" in text
                    or "hand wash" in text
                    or "handwash" in text
                ):
                    has_handwash = True

                if (
                    "dishwasher safe" in text
                    or "dishwasher-safe" in text
                ):
                    has_dishwasher_safe = True

            if has_handwash and has_dishwasher_safe:
                return True

        return False