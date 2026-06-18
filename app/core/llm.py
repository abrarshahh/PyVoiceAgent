import os
import yaml
from langchain_ollama import ChatOllama

# Try importing ChatOpenAI if langchain_openai is installed, otherwise handle gracefully
try:
    from langchain_openai import ChatOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Load config
CONFIG_PATH = "config/config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
else:
    config = {}

def get_llm(json_mode: bool = False):
    provider = config.get("LLM_PROVIDER", "ollama")
    
    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError(
                "langchain-openai is not installed. Please install it using 'pip install langchain-openai'"
            )
        
        model = config.get("LLM_MODEL", "gpt-4o")
        kwargs = {}
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        
        # Ensure api_key is fetched from environment variable loaded via dotenv in app/core/config
        from app.core.config import OPENAI_API_KEY
        return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, **kwargs)
        
    else:
        # Default to Ollama
        model = config.get("LLM_MODEL", "deepseek-r1:8b")
        base_url = config.get("LLM_BASE_URL", "http://localhost:11434")
        
        kwargs = {}
        if json_mode:
            kwargs["format"] = "json"
            
        return ChatOllama(model=model, base_url=base_url, **kwargs)

def get_embeddings():
    provider = config.get("LLM_PROVIDER", "ollama")
    
    if provider == "openai":
        if not HAS_OPENAI:
            raise ImportError(
                "langchain-openai is not installed. Please install it using 'pip install langchain-openai'"
            )
        from langchain_openai import OpenAIEmbeddings
        from app.core.config import OPENAI_API_KEY
        
        model = config.get("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)
    else:
        from langchain_community.embeddings import OllamaEmbeddings as CommunityOllamaEmbeddings
        
        model = config.get("LLM_MODEL", "deepseek-r1:8b")
        base_url = config.get("LLM_BASE_URL", "http://localhost:11434")
        return CommunityOllamaEmbeddings(model=model, base_url=base_url)
