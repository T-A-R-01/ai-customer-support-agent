from app.rag.loader import load_knowledge_base


documents = load_knowledge_base()

print(f"Loaded {len(documents)} documents.\n")

for document in documents:
    print("=" * 60)
    print(f"Source: {document.source}")
    print(f"Metadata: {document.metadata}")
    print(f"Content characters: {len(document.content)}")