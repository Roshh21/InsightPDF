# backend/qa_agent.py
from langchain_core.prompts import ChatPromptTemplate
from .llm_provider import get_llm
from .rag_retriever_agent import answer_with_rag

# 1) Decide / rewrite the question for better retrieval
ROUTE_OR_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query rewriting agent for a Retrieval-Augmented Generation (RAG) system.
Given a user question, rewrite it to maximize retrieval quality over a long PDF.

Guidelines:
- Add important context or synonyms if useful.
- Expand pronouns ("he", "she", "they") into explicit names if mentioned.
- If the question is already clear and specific, return it unchanged.

Return ONLY the rewritten query text, no explanations.
""",
        ),
        ("human", "Original question:\n{question}\n\nRewritten query:"),
    ]
)

# 2) Grade whether the answer is good enough
GRADE_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an answer grading agent.
Given the user's question and an answer, decide if the answer is:

- 'GOOD'  if it is specific, grounded, and fully answers the question.
- 'BAD'   if it is vague, missing key details, or says it doesn't know.

Return EXACTLY one word: GOOD or BAD.
""",
        ),
        ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nGrade (GOOD or BAD):"),
    ]
)

# 3) Refine the question if the first answer was bad
REFINE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query refinement agent.
The previous answer from the RAG system was not good enough.

Given the original question and the weak answer,
produce a NEW, more detailed query that will help retrieve better context from the document.

Make the query:
- More specific (include names, sections, or roles if relevant).
- Explicitly ask for missing details you expect (e.g., background, motivations, key events).

Return ONLY the refined query text, no explanations.
""",
        ),
        (
            "human",
            "Original question:\n{question}\n\nPrevious answer:\n{answer}\n\nRefined query:",
        ),
    ]
)

def _rewrite_query(question: str) -> str:
    llm = get_llm()
    chain = ROUTE_OR_REWRITE_PROMPT | llm
    rewritten = chain.invoke({"question": question})
    return rewritten.strip()

def _grade_answer(question: str, answer: str) -> bool:
    llm = get_llm()
    chain = GRADE_ANSWER_PROMPT | llm
    grade = chain.invoke({"question": question, "answer": answer}).strip().upper()
    return grade == "GOOD"

def _refine_query(question: str, answer: str) -> str:
    llm = get_llm()
    chain = REFINE_QUESTION_PROMPT | llm
    refined = chain.invoke({"question": question, "answer": answer})
    return refined.strip()

def answer_question(question: str):
    """
    Agentic RAG entrypoint for the app:
    - Rewrite query for retrieval
    - Run RAG
    - Grade answer
    - If bad, refine query and retry once
    """
    # 1) Rewrite query for better retrieval
    rewritten_query = _rewrite_query(question)

    # 2) First RAG attempt
    answer, docs = answer_with_rag(rewritten_query)
    if _grade_answer(question, answer):
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        return answer, sources

    # 3) If answer is bad, refine query and try again
    refined_query = _refine_query(question, answer)
    answer2, docs2 = answer_with_rag(refined_query)
    sources2 = [doc.metadata.get("source", "Unknown") for doc in docs2]
    return answer2, sources2