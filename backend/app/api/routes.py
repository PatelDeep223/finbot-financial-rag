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
    from app.db import service as db_service

    stats = pipeline.get_stats()
    db_total = await db_service.get_total_query_count()
    if db_total is not None:
        stats["total_queries_all_time"] = db_total
        db_cache_hits = await db_service.get_total_cache_hits()
        stats["cache_hits_all_time"] = db_cache_hits or 0
    return stats


@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    from app.db import service as db_service

    db_messages = await db_service.get_conversation_history(session_id)
    if db_messages is not None:
        return {"session_id": session_id, "messages": db_messages}
    # Fallback to in-memory
    history = pipeline.conversation_history.get(session_id, [])
    return {"session_id": session_id, "messages": history}


@router.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a session"""
    from app.db import service as db_service

    await db_service.delete_conversation_history(session_id)
    if session_id in pipeline.conversation_history:
        del pipeline.conversation_history[session_id]
    return {"message": f"History cleared for session {session_id}"}


@router.get("/documents")
async def list_documents():
    """List all ingested documents"""
    from app.db import service as db_service

    # Try DB first
    db_docs = await db_service.list_documents()
    if db_docs is not None:
        return {"documents": db_docs, "total": len(db_docs)}

    # Fallback to filesystem
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
