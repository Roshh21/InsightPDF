from fastapi import APIRouter, HTTPException
from ..models.query import QueryRequest, QueryResponse
from ..agents.qa_agent import answer_question

router = APIRouter(prefix="/query", tags=["qa"])

@router.post("", response_model=QueryResponse)
async def query_pdf(payload: QueryRequest):
  if not payload.doc_id:
    raise HTTPException(status_code=400, detail="doc_id is required")

  answer = answer_question(payload.doc_id, payload.question)
  return QueryResponse(text=answer)
