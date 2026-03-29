"""Database service layer with graceful fallback on failure."""

from typing import Optional, List
from sqlalchemy import select, delete, func
from app.db import async_session_factory
from app.models.database import Document, Conversation, QueryLog


async def save_document_record(filename: str, file_size_bytes: int, chunks_created: int):
    try:
        async with async_session_factory() as session:
            session.add(Document(
                filename=filename,
                file_size_bytes=file_size_bytes,
                chunks_created=chunks_created,
            ))
            await session.commit()
    except Exception as e:
        print(f"⚠️ DB: Failed to save document record: {e}")


async def list_documents() -> Optional[List[dict]]:
    """Returns list of document dicts, or None if DB is unreachable."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Document).order_by(Document.uploaded_at.desc())
            )
            docs = result.scalars().all()
            return [
                {
                    "filename": d.filename,
                    "size_kb": round(d.file_size_bytes / 1024, 1),
                    "chunks_created": d.chunks_created,
                    "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                }
                for d in docs
            ]
    except Exception as e:
        print(f"⚠️ DB: Failed to list documents: {e}")
        return None


async def save_conversation_message(
    session_id: str,
    role: str,
    content: str,
    confident: Optional[bool] = None,
    confidence_score: Optional[float] = None,
    sources: Optional[list] = None,
):
    try:
        async with async_session_factory() as session:
            session.add(Conversation(
                session_id=session_id,
                role=role,
                content=content,
                confident=confident,
                confidence_score=confidence_score,
                sources=sources,
            ))
            await session.commit()
    except Exception as e:
        print(f"⚠️ DB: Failed to save conversation: {e}")


async def get_conversation_history(session_id: str) -> Optional[List[dict]]:
    """Returns conversation messages, or None if DB is unreachable."""
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.asc())
            )
            messages = result.scalars().all()
            return [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
    except Exception as e:
        print(f"⚠️ DB: Failed to get conversation history: {e}")
        return None


async def delete_conversation_history(session_id: str):
    try:
        async with async_session_factory() as session:
            await session.execute(
                delete(Conversation).where(Conversation.session_id == session_id)
            )
            await session.commit()
    except Exception as e:
        print(f"⚠️ DB: Failed to delete conversation history: {e}")


async def log_query(
    question: str,
    rewritten_query: Optional[str],
    answer: str,
    confidence_score: float,
    confident: bool,
    from_cache: bool,
    response_time_ms: float,
    user_id: str = "anonymous",
    session_id: Optional[str] = None,
):
    try:
        async with async_session_factory() as session:
            session.add(QueryLog(
                question=question,
                rewritten_query=rewritten_query,
                answer=answer,
                confidence_score=confidence_score,
                confident=confident,
                from_cache=from_cache,
                response_time_ms=response_time_ms,
                user_id=user_id,
                session_id=session_id,
            ))
            await session.commit()
    except Exception as e:
        print(f"⚠️ DB: Failed to log query: {e}")


async def get_total_query_count() -> Optional[int]:
    try:
        async with async_session_factory() as session:
            result = await session.execute(select(func.count(QueryLog.id)))
            return result.scalar()
    except Exception:
        return None


async def get_total_cache_hits() -> Optional[int]:
    try:
        async with async_session_factory() as session:
            result = await session.execute(
                select(func.count(QueryLog.id)).where(QueryLog.from_cache == True)  # noqa: E712
            )
            return result.scalar()
    except Exception:
        return None
