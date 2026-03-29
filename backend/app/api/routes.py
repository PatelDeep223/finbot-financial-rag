import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, UploadResponse, SystemStats
from app.rag.pipeline import pipeline
from app.core.config import settings
from datetime import datetime

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_financial_docs(request: QueryRequest):
    """
    Main RAG query endpoint.
    - Checks semantic cache first
    - Rewrites query for better retrieval  
    - Detects hallucinations
    - Returns answer with sources + confidence score
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if len(request.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")
    
    result = await pipeline.query(
        question=request.question,
        user_id=request.user_id,
        session_id=request.session_id
    )
    
    return QueryResponse(**result)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload financial documents (PDF or TXT).
    Documents are chunked and indexed into FAISS vectorstore.
    """
    allowed_types = ["application/pdf", "text/plain"]
    allowed_extensions = [".pdf", ".txt"]
    
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF and TXT files supported. Got: {file_ext}"
        )
    
    # Save uploaded file
    os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)
    file_path = os.path.join(settings.DOCUMENTS_PATH, file.filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Ingest into vectorstore
    chunks_created = await pipeline.ingest_document(file_path, file.filename)
    
    return UploadResponse(
        message=f"Successfully ingested {file.filename}",
        filename=file.filename,
        chunks_created=chunks_created,
        status="success"
    )


@router.get("/stats")
async def get_system_stats():
    """Get system statistics - queries, cache hit rate, uptime"""
    return pipeline.get_stats()


@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    history = pipeline.conversation_history.get(session_id, [])
    return {"session_id": session_id, "messages": history}


@router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    if session_id in pipeline.conversation_history:
        del pipeline.conversation_history[session_id]
    return {"message": f"History cleared for session {session_id}"}


@router.get("/documents")
async def list_documents():
    """List all ingested documents"""
    docs_path = settings.DOCUMENTS_PATH
    if not os.path.exists(docs_path):
        return {"documents": []}
    
    files = []
    for f in os.listdir(docs_path):
        fp = os.path.join(docs_path, f)
        files.append({
            "filename": f,
            "size_kb": round(os.path.getsize(fp) / 1024, 1),
            "modified": datetime.fromtimestamp(
                os.path.getmtime(fp)
            ).isoformat()
        })
    
    return {"documents": files, "total": len(files)}
