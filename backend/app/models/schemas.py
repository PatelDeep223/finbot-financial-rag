from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ─── AUTH SCHEMAS ────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=6, max_length=128)

class LoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    user: UserResponse


# ─── RAG SCHEMAS ─────────────────────────────────────────────────────────────

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

class EvaluationSample(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class EvaluationRequest(BaseModel):
    samples: List[EvaluationSample]

class SampleScore(BaseModel):
    question: str
    faithfulness: float
    faithfulness_reason: str = ""
    answer_relevancy: float
    answer_relevancy_reason: str = ""
    context_precision: float
    context_precision_reason: str = ""
    context_recall: float
    context_recall_reason: str = ""

class AverageScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

class EvaluationResponse(BaseModel):
    num_samples: int
    average_scores: AverageScores
    per_sample: List[SampleScore]

class SystemStats(BaseModel):
    total_queries: int
    cache_hit_rate: float
    avg_response_time_ms: float
    total_documents: int
    total_chunks: int
    uptime_seconds: float
