# backend/vector_store.py
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from .config import CHROMA_DIR, CHROMA_COLLECTION
from .llm_provider import get_embeddings


def _clear_collection(vs: Chroma):
    """Delete all docs in a collection via Chroma API."""
    try:
        vs._collection.delete(where={})
    except Exception:
        # On very old versions this may not exist; ignore silently
        pass


def build_vector_store(chunks: List[Document]):
    """Index raw text chunks for the current PDF (single collection)."""
    embeddings = get_embeddings()
    vs = Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    _clear_collection(vs)
    vs.add_documents(chunks)
    return vs


def load_vector_store():
    """Load the single collection for retrieval."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
