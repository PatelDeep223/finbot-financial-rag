import os
import shutil
import aiofiles
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models.schemas import QueryRequest, QueryResponse, UploadResponse, EvaluationRequest, EvaluationResponse
from app.rag.pipeline import pipeline
from app.core.config import settings
from app.core.security import limiter, require_auth
from datetime import datetime

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_financial_docs(
    request: Request,
    body: QueryRequest,
    auth: dict = Depends(require_auth),
):
    """
    Main RAG query endpoint.
    Rate limit: 10 req/min. Requires auth (API key or JWT).
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(body.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")

    result = await pipeline.query(
        question=body.question,
        user_id=body.user_id,
        session_id=body.session_id,
    )

    return QueryResponse(**result)


@router.post("/query/stream")
@limiter.limit("10/minute")
async def query_stream(
    request: Request,
    body: QueryRequest,
    auth: dict = Depends(require_auth),
):
    """
    Streaming RAG query endpoint (Server-Sent Events).
    Streams LLM tokens as they generate, then sends sources + confidence.

    SSE events:
      event: meta    → {intent, query_rewritten, from_cache}
      event: token   → {token: "..."}  (many of these)
      event: sources → {sources, confident, confidence_score, response_time_ms}
      event: done    → [DONE]
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(body.question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long (max 1000 chars)")

    async def event_generator():
        async for event in pipeline.query_stream(
            question=body.question,
            user_id=body.user_id,
            session_id=body.session_id,
        ):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@router.post("/upload", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    auth: dict = Depends(require_auth),
):
    """
    Upload financial documents (PDF or TXT).
    Rate limit: 5 req/min. Requires auth.
    """
    allowed_extensions = [".pdf", ".txt"]

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF and TXT files supported. Got: {file_ext}",
        )

    os.makedirs(settings.DOCUMENTS_PATH, exist_ok=True)
    file_path = os.path.join(settings.DOCUMENTS_PATH, file.filename)

    contents = await file.read()
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(contents)

    chunks_created = await pipeline.ingest_document(file_path, file.filename)

    return UploadResponse(
        message=f"Successfully ingested {file.filename}",
        filename=file.filename,
        chunks_created=chunks_created,
        status="success",
    )


@router.get("/stats")
@limiter.limit("30/minute")
async def get_system_stats(
    request: Request,
    auth: dict = Depends(require_auth),
):
    """Get system statistics. Rate limit: 30 req/min."""
    return pipeline.get_stats()


@router.get("/history/{session_id}")
@limiter.limit("30/minute")
async def get_conversation_history(
    request: Request,
    session_id: str,
    auth: dict = Depends(require_auth),
):
    """Get conversation history for a session."""
    history = pipeline.conversation_history.get(session_id, [])
    return {"session_id": session_id, "messages": history}


@router.delete("/history/{session_id}")
@limiter.limit("30/minute")
async def clear_conversation_history(
    request: Request,
    session_id: str,
    auth: dict = Depends(require_auth),
):
    """Clear conversation history for a session."""
    if session_id in pipeline.conversation_history:
        del pipeline.conversation_history[session_id]
    return {"message": f"History cleared for session {session_id}"}


@router.post("/evaluate", response_model=EvaluationResponse)
@limiter.limit("3/minute")
async def evaluate_rag(
    request: Request,
    body: EvaluationRequest,
    auth: dict = Depends(require_auth),
):
    """
    RAGAS-style evaluation endpoint.
    Rate limit: 3 req/min. Requires auth.
    """
    if not body.samples:
        raise HTTPException(status_code=400, detail="No samples provided")
    if len(body.samples) > 20:
        raise HTTPException(status_code=400, detail="Max 20 samples per request")

    from app.services.evaluator import evaluator
    results = await evaluator.evaluate_batch(
        [s.model_dump() for s in body.samples]
    )
    return EvaluationResponse(**results)


@router.get("/documents")
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    auth: dict = Depends(require_auth),
):
    """List all ingested documents."""
    docs_path = settings.DOCUMENTS_PATH
    if not os.path.exists(docs_path):
        return {"documents": []}

    files = []
    for f in os.listdir(docs_path):
        fp = os.path.join(docs_path, f)
        files.append({
            "filename": f,
            "size_kb": round(os.path.getsize(fp) / 1024, 1),
            "modified": datetime.fromtimestamp(os.path.getmtime(fp)).isoformat(),
        })

    return {"documents": files, "total": len(files)}
