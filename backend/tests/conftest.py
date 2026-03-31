import os
import sys
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Ensure backend is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock settings BEFORE importing the app
mock_settings = MagicMock()
mock_settings.OPENAI_API_KEY = "sk-test-fake-key"
mock_settings.OPENAI_MODEL = "gpt-3.5-turbo"
mock_settings.EMBEDDING_MODEL = "text-embedding-ada-002"
mock_settings.REDIS_URL = "redis://localhost:9999"
mock_settings.CACHE_TTL = 3600
mock_settings.DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test"
mock_settings.CHUNK_SIZE = 500
mock_settings.CHUNK_OVERLAP = 50
mock_settings.TOP_K_RESULTS = 5
mock_settings.TOP_K_RETRIEVAL = 20
mock_settings.CONFIDENCE_THRESHOLD = -1.0
mock_settings.TEMPERATURE = 0.0
mock_settings.MAX_CONTEXT_TOKENS = 3000
mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
mock_settings.VECTOR_STORE_PATH = "./data/vectorstore"
mock_settings.DOCUMENTS_PATH = tempfile.mkdtemp()

# Mock DB service
mock_db_service = MagicMock()
mock_db_service.get_conversation_history = AsyncMock(return_value=None)
mock_db_service.delete_conversation_history = AsyncMock()
mock_db_service.list_documents = AsyncMock(return_value=None)
mock_db_service.get_total_query_count = AsyncMock(return_value=None)
mock_db_service.get_total_cache_hits = AsyncMock(return_value=None)
mock_db_service.log_query = AsyncMock()
mock_db_service.save_conversation_message = AsyncMock()
mock_db_service.save_document_record = AsyncMock()

# Mock DB module-level objects
mock_async_engine = MagicMock()
mock_async_engine.dispose = AsyncMock()
mock_init_db = AsyncMock()

# Patch settings, pipeline deps, services, and DB before the app imports them
with patch("app.core.config.settings", mock_settings), \
     patch("app.rag.pipeline.settings", mock_settings), \
     patch("app.rag.pipeline.ChatOpenAI"), \
     patch("app.rag.pipeline.OpenAIEmbeddings"), \
     patch("app.rag.pipeline.redis"), \
     patch("app.services.reranker.settings", mock_settings), \
     patch("app.services.context_builder.settings", mock_settings), \
     patch("app.services.reranker.CrossEncoder"), \
     patch.dict("sys.modules", {"app.db.service": mock_db_service}), \
     patch("app.db.async_engine", mock_async_engine), \
     patch("app.db.init_db", mock_init_db), \
     patch("app.db.async_session_factory", MagicMock()):
    from app.rag.pipeline import FinancialRAGPipeline, pipeline
    from app.main import app

from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    return TestClient(app)


def _make_mock_response(question="test question", from_cache=False, session_id=None):
    return {
        "answer": f"Based on the documents, the answer to '{question}' is 42.",
        "sources": [
            {
                "content": "Revenue was $42 billion in Q3 2024...",
                "source": "earnings.pdf",
                "page": 3,
                "score": 0.91,
            }
        ],
        "confident": True,
        "confidence_score": 0.92,
        "from_cache": from_cache,
        "query_rewritten": "What was the Q3 2024 quarterly revenue?" if not from_cache else None,
        "intent": "factual",
        "response_time_ms": 15.5 if from_cache else 1243.5,
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture()
def mock_pipeline_query(monkeypatch):
    async def fake_query(question, user_id="anonymous", session_id=None):
        pipeline.stats["total_queries"] += 1
        resp = _make_mock_response(question=question, session_id=session_id)
        if session_id:
            if session_id not in pipeline.conversation_history:
                pipeline.conversation_history[session_id] = []
            pipeline.conversation_history[session_id].append(
                {"role": "user", "content": question}
            )
            pipeline.conversation_history[session_id].append(
                {"role": "assistant", "content": resp["answer"]}
            )
        return resp

    monkeypatch.setattr(pipeline, "query", fake_query)
    return fake_query


@pytest.fixture()
def mock_pipeline_query_cached(monkeypatch):
    async def fake_query(question, user_id="anonymous", session_id=None):
        pipeline.stats["total_queries"] += 1
        pipeline.stats["cache_hits"] += 1
        return _make_mock_response(question=question, from_cache=True)

    monkeypatch.setattr(pipeline, "query", fake_query)
    return fake_query


@pytest.fixture()
def mock_pipeline_demo(monkeypatch):
    async def fake_query(question, user_id="anonymous", session_id=None):
        pipeline.stats["total_queries"] += 1
        return pipeline._demo_response(question, __import__("time").time())

    monkeypatch.setattr(pipeline, "query", fake_query)
    return fake_query


@pytest.fixture()
def mock_ingest(monkeypatch):
    async def fake_ingest(file_path, filename):
        return 10

    monkeypatch.setattr(pipeline, "ingest_document", fake_ingest)
    return fake_ingest


@pytest.fixture()
def sample_txt(tmp_path):
    f = tmp_path / "test_report.txt"
    f.write_text("Q3 2024 revenue was $42 billion. Net income increased 15%.")
    return f


@pytest.fixture()
def sample_pdf(tmp_path):
    f = tmp_path / "test_report.pdf"
    f.write_bytes(b"%PDF-1.4 fake pdf content for testing")
    return f


@pytest.fixture(autouse=True)
def reset_pipeline_state():
    pipeline.stats = {
        "total_queries": 0,
        "cache_hits": 0,
        "start_time": __import__("time").time(),
    }
    pipeline.conversation_history = {}
    yield
