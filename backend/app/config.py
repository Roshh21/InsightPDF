import os
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")