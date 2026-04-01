from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = "your-openai-api-key"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour

    # PostgreSQL
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/finbot"

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    TOP_K_RETRIEVAL: int = 20  # candidates before reranking
    CONFIDENCE_THRESHOLD: float = -1.0
    TEMPERATURE: float = 0.0
    MAX_CONTEXT_TOKENS: int = 3000  # token limit for context builder

    # Reranker
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Paths
    VECTOR_STORE_PATH: str = "./data/vectorstore"
    DOCUMENTS_PATH: str = "./data/documents"

    # Security (dev mode: auth skipped when no users in DB)
    JWT_SECRET: str = "finbot-super-secret-change-in-production"

    # LangSmith (Observability)
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "finbot-rag"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"

    class Config:
        env_file = ".env"

settings = Settings()
