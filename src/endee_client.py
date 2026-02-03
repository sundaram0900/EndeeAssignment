"""
Endee Client using the official SDK.
"""
from endee import Endee, Precision

class EndeeClient:
    def __init__(self, base_url="http://localhost:8080", api_key=None):
        if api_key:
            self.client = Endee(api_key)
        else:
            self.client = Endee()
        # Set base URL if non-default
        if base_url != "http://localhost:8080":
            self.client.set_base_url(f"{base_url}/api/v1")
        
        self.index_name = "rag_documents"
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
        self._ensure_index()

    def _ensure_index(self):
        """Create the index if it doesn't exist."""
        try:
            # Try to get the index first
            try:
                self.index = self.client.get_index(name=self.index_name)
                print(f"Using existing index: {self.index_name}")
                return
            except Exception:
                pass  # Index doesn't exist, create it
            
            # Create the index
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                space_type="cosine",
                precision=Precision.INT8D
            )
            print(f"Created index: {self.index_name}")
            self.index = self.client.get_index(name=self.index_name)
        except Exception as e:
            # If we get a conflict, the index exists - try to get it
            if "already exists" in str(e).lower() or "conflict" in str(e).lower():
                try:
                    self.index = self.client.get_index(name=self.index_name)
                    print(f"Got existing index after conflict: {self.index_name}")
                    return
                except Exception as e2:
                    print(f"Failed to get index after conflict: {e2}")
            print(f"Error ensuring index: {e}")
            import traceback
            traceback.print_exc()
            self.index = None

    def check_health(self):
        try:
            indexes = self.client.list_indexes()
            return True
        except Exception:
            return False

    def insert_vectors(self, vectors, documents, metadata=None):
        """
        vectors: list of lists of floats
        documents: list of strings (the text content)
        metadata: list of dicts (optional)
        """
        if not self.index:
            self._ensure_index()
            if not self.index:
                return {"error": "Index not available"}

        items = []
        import uuid
        for i, vec in enumerate(vectors):
            item = {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "meta": {"document": documents[i]}
            }
            if metadata and i < len(metadata):
                item["meta"].update(metadata[i])
            items.append(item)

        try:
            self.index.upsert(items)
            return {"status": "success", "count": len(items)}
        except Exception as e:
            return {"error": str(e)}

    def search(self, query_vector, k=5):
        if not self.index:
            return {"error": "Index not available"}
        
        try:
            results = self.index.query(vector=query_vector, top_k=k)
            return results
        except Exception as e:
            return {"error": str(e)}
