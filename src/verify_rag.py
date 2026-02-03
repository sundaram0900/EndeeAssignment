import os
import sys

from endee_client import EndeeClient
from rag_engine import RAGEngine

def verify():
    print("Initializing components...")
    client = EndeeClient()
    engine = RAGEngine()

    print("Checking Endee Health...")
    is_healthy = client.check_health()
    if is_healthy:
        print("Endee is reachable.")
    else:
        print("ERROR: Endee is not reachable at http://localhost:8080.")
        return

    text_to_ingest = "The quick brown fox jumps over the lazy dog. Programming is fun."
    print(f"Ingesting text: '{text_to_ingest}'")

    chunks = engine.chunk_text(text_to_ingest)
    embeddings = engine.generate_embeddings(chunks)
    
    # Ingest
    print("Inserting vectors...")
    try:
        res = client.insert_vectors(embeddings, chunks)
        print(f"Insert response: {res}")
        if "error" in res:
            print("Insert failed!")
            return
    except Exception as e:
        print(f"Insert failed: {e}")
        return

    # Search
    query = "brown fox"
    print(f"Searching for: '{query}'")
    query_embedding = engine.generate_embeddings([query])[0]
    
    try:
        search_res = client.search(query_embedding, k=2)
        print("Search Results:")
        print(search_res)
        
        # Check if results contain expected content
        if search_res and isinstance(search_res, list) and len(search_res) > 0:
            print("\n✓ SUCCESS: Search returned results!")
        else:
            print("\n⚠ WARNING: Search returned no results or unexpected format.")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    verify()
