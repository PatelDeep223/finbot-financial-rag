from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = "your-openai-api-key"
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour
    
    # PostgreSQL
    DATABASE_URL: str = "postgresql://finbot:finbot123@localhost:5432/finbot"
    
    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    CONFIDENCE_THRESHOLD: float = -1.0
    TEMPERATURE: float = 0.0  # Zero for anti-hallucination
    
    # Paths
    VECTOR_STORE_PATH: str = "./data/vectorstore"
    DOCUMENTS_PATH: str = "./data/documents"
    
    class Config:
        env_file = ".env"

settings = Settings()
