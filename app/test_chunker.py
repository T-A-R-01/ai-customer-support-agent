from app.rag.loader import load_knowledge_base
from app.rag.chunker import chunk_documents


documents = load_knowledge_base()

chunks = chunk_documents(documents)

print(f"Loaded documents: {len(documents)}")
print(f"Created chunks: {len(chunks)}")
print()

for chunk in chunks[:10]:
    print("=" * 70)
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Source: {chunk.source}")
    print(f"Heading: {chunk.heading}")
    print(f"Metadata: {chunk.metadata}")
    print(f"Text:\n{chunk.text[:500]}")