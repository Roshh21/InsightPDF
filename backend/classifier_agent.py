# backend/classifier_agent.py
from langchain_core.prompts import ChatPromptTemplate
from .llm_provider import get_llm


CLASSIFICATION_SYSTEM_PROMPT = """
You are a document classifier. Given the text of a PDF (or a large sample),
classify it into exactly one of the following types:
- Research Paper
- Novel / Literature
- Study Material / Notes
- Technical Documentation
- Business Report

Answer ONLY with the type.
"""

def classify_document(sample_text: str) -> str:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLASSIFICATION_SYSTEM_PROMPT),
            ("human", "Here is a sample of the document:\n\n{sample}"),
        ]
    )
    chain = prompt | llm
    result = chain.invoke({"sample": sample_text[:4000]})
    return result.strip()
