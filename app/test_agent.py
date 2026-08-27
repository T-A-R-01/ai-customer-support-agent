from app.agent import SupportAgent


print("Starting support agent...\n")

agent = SupportAgent()


queries = [
    "How long do I have to return an item?",
    "What is the weather on Mars tomorrow?",
    "Where is my order ORD-1001?",
    "Can you check order ord-1001?",
    "Where is my order?",
]


for query in queries:

    print("\n========================================")
    print(f"USER: {query}")
    print("========================================")

    answer = agent.answer(query)

    print("\nASSISTANT:")
    print(answer)