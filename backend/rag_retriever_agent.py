from langchain_core.prompts import ChatPromptTemplate
from .llm_provider import get_llm
from .vector_store import load_raw_vector_store, load_summary_vector_store
from langchain_core.documents import Document
from typing import List, Tuple

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval router agent.

Decide which index is best for answering the user's question:

- "RAW": use fine-grained raw chunks (good for specific details, quotes, formulas).
- "SUMMARY": use section summaries only (good for high-level overviews, big-picture questions).
- "BOTH": use both indexes and combine their context.

Return EXACTLY one word: RAW, SUMMARY, or BOTH.
""",
        ),
        ("human", "Question:\n{question}\n\nYour choice (RAW, SUMMARY, or BOTH):"),
    ]
)
def _route_index(question: str) -> str:
    llm = get_llm()
    chain = ROUTER_PROMPT | llm
    choice = chain.invoke({"question": question}).strip().upper()
    if choice not in {"RAW", "SUMMARY", "BOTH"}:
        return "RAW"
    return choice

def retrieve_context_agentic(query: str, k_raw: int = 20, k_summary: int = 10) -> List[Document]:
    choice = _route_index(query)

    docs: List[Document] = []
    if choice in {"RAW", "BOTH"}:
        raw_vs = load_raw_vector_store()
        raw_retriever = raw_vs.as_retriever(search_kwargs={"k": k_raw})
        docs.extend(raw_retriever.invoke(query))

    if choice in {"SUMMARY", "BOTH"}:
        sum_vs = load_summary_vector_store()
        sum_retriever = sum_vs.as_retriever(search_kwargs={"k": k_summary})
        docs.extend(sum_retriever.invoke(query))

    return docs

def answer_with_rag(query: str) -> Tuple[str, List[Document]]:
    llm = get_llm()
    docs = retrieve_context_agentic(query)
    context = "\n\n".join(d.page_content for d in docs)

    RAG_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about a specific PDF.
Use ONLY the provided context (which may come from raw text or section summaries).
Combine information from all chunks. If the answer is not in the context, say you don't know.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context[:8000], "question": query})
    return answer, docs
