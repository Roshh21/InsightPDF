from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
  doc_id: str
  question: str

class QueryResponse(BaseModel):
  text: str

class QuizRequest(BaseModel):
  doc_id: str
  num_questions: int | None = 5

class QuizQuestion(BaseModel):
  type: str
  prompt: str
  options: List[str]
  answer: str

class QuizResponse(BaseModel):
  questions: List[QuizQuestion]
