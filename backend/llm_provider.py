# backend/llm_provider.py
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from .config import OLLAMA_LLM_MODEL, EMBEDDING_MODEL

def get_llm():
    # Ollama must be running: `ollama serve`
    return Ollama(model=OLLAMA_LLM_MODEL)

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
