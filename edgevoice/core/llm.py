# edgevoice/core/llm.py

def get_llm(json_mode: bool = False):
    raise NotImplementedError("LLM integration is not implemented yet. Configuration lands in Phase 1.")

def get_embeddings():
    raise NotImplementedError("LLM embeddings are not implemented yet. Configuration lands in Phase 1.")

class LocalLLM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("LocalLLM lands in Phase 1.")

class CloudLLM:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("CloudLLM lands in Phase 1.")

class HybridRouter:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HybridRouter lands in Phase 1.")
