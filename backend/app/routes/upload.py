import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ..services.rag_pipeline import ingest_pdf
from ..models.document import UploadResponse, SummaryResponse, SummarySection

router = APIRouter(prefix="/upload", tags=["upload"])

@router.post("", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
  if file.content_type != "application/pdf":
    raise HTTPException(status_code=400, detail="Only PDF files are supported")

  pdf_bytes = await file.read()
  doc_id = str(uuid.uuid4())

  summary_dict, _ = ingest_pdf(doc_id, pdf_bytes)

  sections = [
    SummarySection(title=s["title"], points=s["points"])
    for s in summary_dict.get("sections", [])
  ]
  summary = SummaryResponse(
    docId=doc_id,
    documentType=summary_dict.get("documentType", "other"),
    sections=sections,
  )

  return UploadResponse(docId=doc_id, summary=summary)
