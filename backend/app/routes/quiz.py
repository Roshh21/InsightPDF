from fastapi import APIRouter, HTTPException
from ..models.query import QuizRequest, QuizResponse, QuizQuestion
from ..agents.quiz_agent import generate_quiz

router = APIRouter(prefix="/quiz", tags=["quiz"])

@router.post("", response_model=QuizResponse)
async def quiz_from_pdf(payload: QuizRequest):
  if not payload.doc_id:
    raise HTTPException(status_code=400, detail="doc_id is required")

  raw_quiz = generate_quiz(payload.doc_id, payload.num_questions or 5)
  questions_data = raw_quiz.get("questions", [])

  questions = [
    QuizQuestion(
      type=q.get("type", "short"),
      prompt=q.get("prompt", ""),
      options=q.get("options", []) or [],
      answer=q.get("answer", ""),
    )
    for q in questions_data
  ]

  return QuizResponse(questions=questions)
