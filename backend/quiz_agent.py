# backend/quiz_agent.py
from langchain_core.prompts import ChatPromptTemplate  # or langchain.prompts if on old version
from .llm_provider import get_llm
from .rag_retriever_agent import answer_with_rag

QUIZ_SYSTEM_PROMPT = """
You are a quiz generator. Using ONLY the provided PDF context, create a quiz.

Requirements:
- Include a mix of:
  - 3–5 multiple-choice questions (MCQs) with 4 options each and clearly marked correct answer.
  - 3 True/False questions.
  - 3 short-answer questions.
- Vary difficulty from easy to hard.
- Base everything strictly on the document.
"""

def generate_quiz_from_query(query: str = "Create a quiz based on the whole document."):
    answer, docs = answer_with_rag(query)
    context = "\n\n".join(d.page_content for d in docs)

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QUIZ_SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nGenerate the quiz now."),
        ]
    )
    chain = prompt | llm
    quiz = chain.invoke({"context": context[:8000]})
    return quiz