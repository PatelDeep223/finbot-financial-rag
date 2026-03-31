from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class QueryRequest(BaseModel):
    question: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None

class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    confident: bool
    confidence_score: float
    from_cache: bool
    query_rewritten: Optional[str] = None
    intent: Optional[str] = None
    response_time_ms: float
    timestamp: datetime = datetime.now()

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int
    status: str

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    confident: Optional[bool] = None
    sources: Optional[List[SourceDocument]] = None

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime = datetime.now()

class SystemStats(BaseModel):
    total_queries: int
    cache_hit_rate: float
    avg_response_time_ms: float
    total_documents: int
    total_chunks: int
    uptime_seconds: float
