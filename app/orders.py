import json
import re
from pathlib import Path
from typing import Any


ORDERS_FILE = Path("data/orders.json")


class OrderLookup:
    """Safe lookup service for customer order information."""

    # Only customer-safe fields may leave the lookup layer.
    ALLOWED_FIELDS = {
        "order_id",
        "status",
        "order_date",
        "items",
        "total",
        "currency",
        "shipping_status",
        "estimated_delivery",
        "carrier",
    }

    def __init__(self, orders_file: Path = ORDERS_FILE):
        self.orders_file = orders_file
        self.orders = self._load_orders()

    def _load_orders(self) -> dict[str, dict[str, Any]]:
        """Load orders from the JSON file."""

        if not self.orders_file.exists():
            raise FileNotFoundError(
                f"Orders file not found: {self.orders_file}"
            )

        with self.orders_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        # Support:
        # {"ORD-123": {...}}
        # or:
        # {"orders": [{...}]}
        # or:
        # [{...}]

        if isinstance(data, dict):

            if (
                "orders" in data
                and isinstance(data["orders"], list)
            ):
                data = data["orders"]

            else:
                return {
                    str(order_id).upper(): order
                    for order_id, order in data.items()
                    if isinstance(order, dict)
                }

        if isinstance(data, list):

            orders = {}

            for order in data:

                if not isinstance(order, dict):
                    continue

                order_id = order.get("order_id")

                if order_id:
                    orders[str(order_id).upper()] = order

            return orders

        raise ValueError(
            "Unsupported orders.json structure."
        )

    def _normalize_order_id(
        self,
        order_id: str,
    ) -> str:
        """Normalize and validate an order ID."""

        if not isinstance(order_id, str):
            return ""

        normalized = order_id.strip().upper()

        if len(normalized) > 50:
            return ""

        if not re.fullmatch(
            r"[A-Z0-9_-]+",
            normalized,
        ):
            return ""

        return normalized

    def lookup(
        self,
        order_id: str,
    ) -> dict[str, Any]:
        """
        Look up an order and return only
        customer-safe fields.
        """

        normalized_id = self._normalize_order_id(
            order_id
        )

        if not normalized_id:
            return {
                "found": False,
                "error": "Invalid order ID.",
            }

        order = self.orders.get(normalized_id)

        if order is None:
            return {
                "found": False,
                "error": "Order not found.",
            }

        safe_order = {
            key: value
            for key, value in order.items()
            if key in self.ALLOWED_FIELDS
        }

        safe_order["found"] = True

        return safe_order