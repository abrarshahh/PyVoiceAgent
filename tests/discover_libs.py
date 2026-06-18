import sys
import os

try:
    from qdrant_client import QdrantClient
    client = QdrantClient(":memory:")
    print(f"Qdrant instance has search: {'search' in dir(client)}")
    print(f"Qdrant instance has query_points: {'query_points' in dir(client)}")
except Exception as e:
    print(f"Qdrant error: {e}")

try:
    from chatterbox.tts import ChatterboxTTS
    import inspect
    tts = ChatterboxTTS()
    print(f"ChatterboxTTS.generate signature: {inspect.signature(tts.generate)}")
except Exception as e:
    print(f"Chatterbox error: {e}")
