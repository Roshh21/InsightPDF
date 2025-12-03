from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from .llm_provider import get_llm
from .vector_store import load_vector_store

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a retrieval router agent.

For now there is only ONE index of raw chunks for the PDF,
but you should still analyze the question for difficulty and specificity.
In future, you might choose between RAW and SUMMARY indexes.

For now, always answer with "RAW".
""",
        ),
        ("human", "Question:\n{question}\n\nYour choice (RAW):"),
    ]
)

def _route_index(question: str) -> str:
    llm = get_llm()
    chain = ROUTER_PROMPT | llm
    choice = str(chain.invoke({"question": question})).strip().upper()
    # Since we only have one index, force RAW as a safe default.
    if choice not in {"RAW"}:
        return "RAW"
    return choice

def retrieve_context_agentic(query: str, k: int = 25) -> List[Document]:
    """
    Agentic-style retrieval wrapper, currently over a single raw index.
    Router is kept for future extension, but always ends up using RAW.
    """
    _ = _route_index(query)  # kept for extensibility
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs

def answer_with_rag(query: str) -> Tuple[str, List[Document]]:
    llm = get_llm()
    docs = retrieve_context_agentic(query)
    context = "\n\n".join(d.page_content for d in docs)

    RAG_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions about a specific PDF.
Use ONLY the provided context (from the indexed chunks).
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