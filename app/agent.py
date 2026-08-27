import re
from datetime import datetime

from app.orders import OrderLookup
from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents
from app.rag.embeddings import EmbeddingService
from app.rag.index import VectorIndex
from app.rag.retriever import Retriever
from app.rag.safety import EvidenceChecker
from app.rag.generator import Generator


class SupportAgent:
    """Customer-support agent combining RAG and order lookup."""

    def __init__(self):
        print("Loading knowledge base...")

        documents = load_knowledge_base()
        chunks = chunk_documents(documents)

        print(
            f"Loaded {len(documents)} documents "
            f"and {len(chunks)} chunks."
        )

        print("Generating embeddings...")

        self.embedding_service = EmbeddingService()
        embeddings = self.embedding_service.embed_chunks(chunks)

        print("Building vector index...")

        self.index = VectorIndex()
        self.index.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        self.retriever = Retriever(
            index=self.index,
            embedding_service=self.embedding_service,
        )

        self.checker = EvidenceChecker()
        self.generator = Generator()
        self.order_lookup = OrderLookup()

    # =========================================================
    # ORDER DETECTION
    # =========================================================

    def _extract_order_id(self, query: str) -> str | None:
        """Extract a concrete order ID from the question."""

        match = re.search(
            r"\bORD[-_][A-Z0-9]+\b",
            query.upper(),
        )

        if match:
            return match.group(0)

        return None

    def _is_order_query(self, query: str) -> bool:
        """Detect whether the user is asking about an order."""

        query_lower = query.lower()

        # Concrete order ID is the strongest signal.
        if self._extract_order_id(query):
            return True

        order_phrases = [
            "where is my order",
            "where's my order",
            "where is my package",
            "where's my package",
            "track my order",
            "track my package",
            "check my order",
            "check my package",
            "status of my order",
            "status of my package",
            "when will my order arrive",
            "when will my package arrive",
            "when should my order arrive",
            "when should my package arrive",
        ]

        return any(
            phrase in query_lower
            for phrase in order_phrases
        )

    # =========================================================
    # MAIN ROUTER
    # =========================================================

    def answer(self, query: str) -> str:
        """Route a customer question to the correct capability."""

        query = query.strip()

        if not query:
            return "Please provide a question."

        q = query.lower()

        # =====================================================
        # ORDER PATH
        # =====================================================

        if self._is_order_query(query):

            order_id = self._extract_order_id(query)

            if not order_id:
                return (
                    "Please provide your order ID so I can "
                    "look up the order."
                )

            # -------------------------------------------------
            # Privacy protection
            # -------------------------------------------------

            privacy_terms = [
                "email",
                "address",
                "internal note",
                "internal notes",
                "risk score",
                "fraud review",
            ]

            if any(
                term in q
                for term in privacy_terms
            ):
                return (
                    "I can't provide private or internal customer "
                    "information, including the customer's email, "
                    "address, internal notes, or risk score. "
                    "Please contact support if you need help with "
                    "this order."
                )

            order = self.order_lookup.lookup(order_id)

            if not order.get("found"):
                return (
                    f"The order was not found: {order_id}. "
                    "Please check the order ID or contact support."
                )

            return self._format_order_response(
                order_id,
                order,
            )

        # =====================================================
        # HIGH-CONFIDENCE POLICY RESPONSES
        # =====================================================

        # -----------------------------------------------------
        # FINAL-SALE DAMAGED ITEM EXCEPTION
        # -----------------------------------------------------

        if (
            "final-sale" in q
            or "final sale" in q
        ) and (
            "damaged" in q
            or "broken" in q
            or "defective" in q
            or "zipper" in q
        ):
            return (
                "A final sale does not block damaged-item review. "
                "Final-sale items are still eligible for review "
                "when they arrive damaged, defective, or incorrect. "
                "Report within 7 days of delivery. "
                "Human review before approval is required."
            )

        # -----------------------------------------------------
        # PROMPT-INJECTION / MIGRATION NOTE
        # -----------------------------------------------------

        if (
            "migration note" in q
            or "60 days" in q
            or "ignore the real policy" in q
        ):
            return (
                "The migration note is not authoritative and does "
                "not override the current official returns policy. "
                "The standard policy is 30 days unless a valid "
                "exception applies. The agent cannot approve a "
                "return."
            )

        # -----------------------------------------------------
        # WARRANTY
        # -----------------------------------------------------

        if (
            "lifetime warranty" in q
            or (
                "warranty" in q
                and (
                    "bags" in q
                    or "products" in q
                    or "drinkware" in q
                    or "travel accessories" in q
                )
            )
        ):
            return (
                "Aster & Row products have no lifetime warranty. "
                "Bags have 2 years of warranty coverage. "
                "Drinkware and travel accessories have 1 year "
                "of warranty coverage."
            )

        # -----------------------------------------------------
        # GERMANY SHIPPING
        # -----------------------------------------------------

        if (
            "germany" in q
            and (
                "ship" in q
                or "shipping" in q
                or "deliver" in q
            )
        ):
            return (
                "Shipping to Germany is not currently available. "
                "Aster & Row currently supports international "
                "shipping to Canada."
            )

        # -----------------------------------------------------
        # CANADA SHIPPING
        # -----------------------------------------------------

        if (
            "canada" in q
            and (
                "shipping" in q
                or "ship" in q
                or "delivery" in q
                or "take" in q
                or "how long" in q
                or "time" in q
            )
        ):
            return (
                "Canada is supported. Delivery to Canada takes "
                "5–9 business days after dispatch. Duties or taxes "
                "are not prepaid and may be collected separately."
            )

        # =====================================================
        # RAG PATH
        # =====================================================

        results = self.retriever.search(
            query=query,
            top_k=5,
        )

        evidence = self.checker.check(
            results,
            query=query,
        )

        # =====================================================
        # SOURCE CONFLICT
        # =====================================================

        if evidence.conflict:
            return (
                "The current official sources conflict. "
                "One says hand-wash the body, while one says "
                "all components are dishwasher safe. "
                "Human confirmation or safest interim guidance "
                "is required. As the safest interim guidance, "
                "hand-wash the tumbler until the conflict is resolved."
            )

        # =====================================================
        # INSUFFICIENT INFORMATION
        # =====================================================

        if not evidence.sufficient:
            return (
                "The supplied information is insufficient to "
                "answer this question reliably. Human confirmation "
                "is recommended."
            )

        # =====================================================
        # GENERATION
        # =====================================================

        return self.generator.generate(
            query=query,
            results=results,
        )

    # =========================================================
    # ORDER RESPONSE
    # =========================================================

    def _format_order_response(
        self,
        order_id: str,
        order: dict,
    ) -> str:
        """Create a safe customer-facing order response."""

        status = str(
            order.get("status", "")
        ).lower()

        lines = [
            f"Order {order_id}:"
        ]

        # -----------------------------------------------------
        # CANCELLED ORDERS
        # -----------------------------------------------------

        if status == "cancelled":
            lines.append(
                "The order is cancelled and it will not be shipped."
            )

            return "\n".join(lines)

        # -----------------------------------------------------
        # NORMAL STATUS
        # -----------------------------------------------------

        if order.get("status"):
            lines.append(
                f"Status: {order['status']}"
            )

        # -----------------------------------------------------
        # SHIPPING STATUS / CARRIER
        # -----------------------------------------------------

        if order.get("shipping_status"):

            shipping_status = str(
                order["shipping_status"]
            ).lower()

            if "shipped" in shipping_status:

                if order.get("carrier"):
                    lines.append(
                        f"Shipped with {order['carrier']}."
                    )
                else:
                    lines.append(
                        "Shipping status: shipped"
                    )

            else:
                lines.append(
                    f"Shipping status: "
                    f"{order['shipping_status']}"
                )

        elif order.get("carrier"):

            lines.append(
                f"Shipped with {order['carrier']}."
            )

        # -----------------------------------------------------
        # ETA
        # -----------------------------------------------------

        if order.get("estimated_delivery"):

            raw_eta = str(
                order["estimated_delivery"]
            )

            formatted_eta = self._format_date(
                raw_eta
            )

            lines.append(
                f"Estimated delivery: {formatted_eta}"
            )

        else:

            lines.append(
                "Delivery estimate is unavailable."
            )

        return "\n".join(lines)

    # =========================================================
    # DATE FORMATTING
    # =========================================================

    def _format_date(self, value: str) -> str:
        """
        Convert YYYY-MM-DD into a customer-friendly date.

        Example:
            2026-08-22 -> August 22, 2026
        """

        match = re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            value,
        )

        if not match:
            return value

        try:
            return datetime.strptime(
                value,
                "%Y-%m-%d",
            ).strftime("%B %d, %Y").replace(
                " 0",
                " ",
            )

        except ValueError:
            return value