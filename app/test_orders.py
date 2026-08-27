from app.orders import OrderLookup


print("Loading orders...")

lookup = OrderLookup()

print(f"Orders loaded: {len(lookup.orders)}")


test_ids = [
    "ORD-1001",
    "ord-1001",
    "INVALID-ORDER",
    "",
]


for order_id in test_ids:

    print("\n========================================")
    print(f"ORDER ID: {order_id!r}")
    print("========================================")

    result = lookup.lookup(order_id)

    print(result)