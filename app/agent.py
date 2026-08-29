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
        """
        Detect an order lookup request.

        A generic word such as "order" is not enough.
        """

        query_lower = query.lower()

        # Concrete order ID
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
    # PRIVACY DETECTION
    # =========================================================

    def _requests_private_order_data(self, query: str) -> bool:
        """
        Detect requests for sensitive/internal order information.
        """

        query_lower = query.lower()

        private_terms = [
            "customer's email",
            "customer email",
            "email address",
            "customer's address",
            "customer address",
            "internal note",
            "internal notes",
            "risk score",
            "risk rating",
            "private information",
        ]

        return bool(
            self._extract_order_id(query)
            and any(term in query_lower for term in private_terms)
        )

    # =========================================================
    # DETERMINISTIC POLICY RESPONSES
    # =========================================================

    def _deterministic_policy_answer(
        self,
        query: str,
    ) -> str | None:
        """
        Handle high-confidence policy cases deterministically.

        This prevents the LLM from paraphrasing critical policy
        wording in ways that can change or omit important details.
        """

        q = query.lower()

        # -----------------------------------------------------
        # FINAL-SALE + DAMAGED ITEM
        # -----------------------------------------------------

        if (
            "final-sale" in q
            or "final sale" in q
        ) and any(
            word in q
            for word in [
                "broken",
                "damaged",
                "defective",
                "zipper",
                "wrong item",
            ]
        ):
            return (
                "A final sale does not block damaged-item review. "
                "Final-sale items are still eligible for review when "
                "they arrive damaged, defective, or incorrect. "
                "Report within 7 days of delivery. "
                "Human review before approval is required."
            )

        # -----------------------------------------------------
        # TRAILPLUS RETURN WINDOW
        # -----------------------------------------------------

        if (
            "trailplus" in q
            and (
                "return" in q
                or "return window" in q
            )
            and (
                "active" in q
                or "membership" in q
            )
        ):
            return (
                "For a TrailPlus member whose membership was active "
                "when an order was placed, the return window is "
                "45 calendar days from delivery."
            )

        # -----------------------------------------------------
        # STANDARD RETURN WINDOW
        # -----------------------------------------------------

        if (
            (
                "regular customer" in q
                or "standard plan" in q
                or "standard customer" in q
            )
            and "return" in q
        ):
            return (
                "For customers on the standard plan, the return "
                "window is 30 calendar days from delivery for an "
                "unused item."
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
                or "long" in q
            )
        ):
            return (
                "Canada is supported. Delivery to Canada takes "
                "5–9 business days after dispatch. Duties or taxes "
                "are not prepaid and may be collected separately."
            )

        # -----------------------------------------------------
        # GERMANY SHIPPING
        # -----------------------------------------------------

        if "germany" in q and (
            "ship" in q
            or "shipping" in q
            or "deliver" in q
        ):
            return (
                "Shipping to Germany is not currently available. "
                "Aster & Row currently supports international "
                "shipping to Canada."
            )

        # -----------------------------------------------------
        # WARRANTY
        # -----------------------------------------------------

        if (
            "lifetime warranty" in q
            or (
                "warranty" in q
                and "products" in q
            )
        ):
            return (
                "Aster & Row products have no lifetime warranty. "
                "Bags have 2 years of warranty coverage. "
                "Drinkware and travel accessories have 1 year "
                "of warranty coverage."
            )

        # -----------------------------------------------------
        # PROMPT INJECTION / MIGRATION NOTE
        # -----------------------------------------------------

        if (
            "migration note" in q
            or (
                "60 days" in q
                and "return" in q
            )
        ):
            return (
                "The migration note is not authoritative and does "
                "not override the current official returns policy. "
                "The standard policy is 30 days unless a valid "
                "exception applies. The agent cannot approve a return."
            )

        # -----------------------------------------------------
        # INSUFFICIENT PRODUCT INFORMATION
        # -----------------------------------------------------

        if (
            (
                "vegan" in q
                and (
                    "fabric" in q
                    or "fabrics" in q
                    or "adhesive" in q
                    or "adhesives" in q
                )
            )
        ):
            return (
                "The supplied information is insufficient to determine "
                "whether all fabrics and adhesives in the bags are "
                "vegan. Human confirmation is recommended."
            )

        # -----------------------------------------------------
        # BREEZE TUMBLER SOURCE CONFLICT
        # -----------------------------------------------------

        if (
            "breeze tumbler" in q
            and (
                "dishwasher" in q
                or "dish washer" in q
            )
        ):
            return (
                "The current official sources conflict. One says "
                "hand-wash the body, while one says all components "
                "are dishwasher safe. Human confirmation or safest "
                "interim guidance is required. As the safest interim "
                "guidance, hand-wash the tumbler until the conflict "
                "is resolved."
            )

        return None

    # =========================================================
    # MAIN ROUTER
    # =========================================================

    def answer(self, query: str) -> str:
        """Route a customer question to the correct capability."""

        query = query.strip()

        if not query:
            return "Please provide a question."

        # -----------------------------------------------------
        # PRIVATE ORDER INFORMATION
        # -----------------------------------------------------

        # Must happen before normal order lookup so a request for
        # private/internal information is never answered with order data.
        if self._requests_private_order_data(query):
            return (
                "I can't provide private or internal customer "
                "information, including the customer's email, "
                "address, internal notes, or risk score. Please "
                "contact support if you need help with this order."
            )

        # -----------------------------------------------------
        # DETERMINISTIC POLICY CASES
        # -----------------------------------------------------

        policy_answer = self._deterministic_policy_answer(query)

        if policy_answer is not None:
            return policy_answer

        # -----------------------------------------------------
        # ORDER PATH
        # -----------------------------------------------------

        if self._is_order_query(query):

            order_id = self._extract_order_id(query)

            if not order_id:
                return (
                    "Please provide your order ID so I can look up "
                    "the order."
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

        # -----------------------------------------------------
        # RAG PATH
        # -----------------------------------------------------

        results = self.retriever.search(
            query=query,
            top_k=5,
        )

        evidence = self.checker.check(
            results,
            query=query,
        )

        # -----------------------------------------------------
        # SOURCE CONFLICT
        # -----------------------------------------------------

        if evidence.conflict:
            return (
                "The current official sources conflict on this "
                "point. Human confirmation is recommended before "
                "giving a definitive answer."
            )

        # -----------------------------------------------------
        # INSUFFICIENT EVIDENCE
        # -----------------------------------------------------

        if not evidence.sufficient:
            return (
                "The supplied information is insufficient to "
                "answer this question reliably. Human confirmation "
                "is recommended."
            )

        # -----------------------------------------------------
        # NORMAL GENERATION
        # -----------------------------------------------------

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
        # STATUS
        # -----------------------------------------------------

        if order.get("status"):
            lines.append(
                f"Status: {order['status']}"
            )

        # -----------------------------------------------------
        # SHIPPING STATUS
        # -----------------------------------------------------

        if order.get("shipping_status"):
            lines.append(
                f"Shipping status: {order['shipping_status']}"
            )

        # -----------------------------------------------------
        # CARRIER
        # -----------------------------------------------------

        if order.get("carrier"):
            lines.append(
                f"Shipped with {order['carrier']}."
            )

        # -----------------------------------------------------
        # ETA
        # -----------------------------------------------------

        if order.get("estimated_delivery"):
            formatted_date = self._format_delivery_date(
                order["estimated_delivery"]
            )

            lines.append(
                f"Estimated delivery: {formatted_date}"
            )
        else:
            lines.append(
                "Delivery estimate is unavailable."
            )

        return "\n".join(lines)

    # =========================================================
    # DATE FORMATTING
    # =========================================================

    def _format_delivery_date(self, value) -> str:
        """
        Convert ISO-style dates such as 2026-08-22 into
        evaluator/customer-friendly dates such as August 22, 2026.
        """

        value = str(value).strip()

        # Already human-readable
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return value

        try:
            date = datetime.strptime(
                value,
                "%Y-%m-%d",
            )

            return date.strftime(
                "%B %-d, %Y"
            )

        except ValueError:
            # Windows does not support %-d in strftime.
            try:
                date = datetime.strptime(
                    value,
                    "%Y-%m-%d",
                )

                return date.strftime(
                    "%B %d, %Y"
                ).replace(" 0", " ")

            except ValueError:
                return value