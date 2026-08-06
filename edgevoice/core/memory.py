import os
import time
from typing import List, Dict, Any, Optional
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False
import yaml
from edgevoice.core.llm import get_embeddings

# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class MemoryAgent:
    def __init__(self):
        if not HAS_QDRANT:
            print("WARNING: qdrant-client is not installed. Memory agent will be disabled.")
            self.client = None
            return
        self.qdrant_url = os.getenv("QDRANT_URL", config.get("QDRANT_URL"))
        self.qdrant_key = os.getenv("QDRANT_API_KEY", config.get("QDRANT_API_KEY"))
        self.collection_name = config.get("QDRANT_COLLECTION", "pyvoiceagent_memory")
        
        if not self.qdrant_url:
            print("WARNING: QDRANT_URL not set. Memory agent will be disabled.")
            self.client = None
            return

        print(f"Connecting to Qdrant at {self.qdrant_url}...")
        # If url is HTTPS and cloud-based, default port should be 443
        port = None
        if self.qdrant_url.startswith("https://") and ":" not in self.qdrant_url.replace("https://", ""):
            port = 443
            
        self.client = QdrantClient(url=self.qdrant_url, port=port, api_key=self.qdrant_key)
        
        # Initialize Embeddings
        try:
            self.embeddings = get_embeddings()
        except NotImplementedError:
            print("WARNING: Embeddings model is not implemented yet. Memory agent will be disabled.")
            self.client = None
            return
        
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                print(f"Creating collection {self.collection_name}...")
                # Assuming 4096 dim for deepseek/llama, need to verify model dimension
                # Quick check: embed "hello" and check len
                test_embed = self.embeddings.embed_query("hello")
                vector_size = len(test_embed)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                print(f"Collection created with vector size {vector_size}.")
        except Exception as e:
            print(f"Error ensuring collection: {e}")

    def add_memory(self, text: str, metadata: Dict[str, Any] = None):
        if not self.client:
            return
            
        try:
            vector = self.embeddings.embed_query(text)
            
            point = PointStruct(
                id=int(time.time() * 1000), # Simple ID generation
                vector=vector,
                payload={"text": text, **(metadata or {})}
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            print(f"Memory added: {text[:50]}...")
        except Exception as e:
            print(f"Error adding memory: {e}")

    def retrieve_memory(self, query: str, limit: int = 3) -> List[str]:
        if not self.client:
            return []
            
        try:
            vector = self.embeddings.embed_query(query)
            
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit
            ).points
            
            memories = [hit.payload.get("text", "") for hit in results]
            print(f"Retrieved {len(memories)} memories.")
            return memories
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return []

if __name__ == "__main__":
    agent = MemoryAgent()
    # agent.add_memory("User prefers responses in bullet points.", {"type": "preference"})
    # print(agent.retrieve_memory("How should I format the response?"))
