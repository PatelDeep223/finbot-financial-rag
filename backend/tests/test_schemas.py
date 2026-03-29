from datetime import datetime
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceDocument,
    UploadResponse,
    ChatMessage,
    ConversationHistory,
)


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(question="test")
        assert req.user_id == "anonymous"
        assert req.session_id is None

    def test_custom_values(self):
        req = QueryRequest(question="revenue?", user_id="u1", session_id="s1")
        assert req.question == "revenue?"
        assert req.user_id == "u1"
        assert req.session_id == "s1"


class TestSourceDocument:
    def test_required_fields(self):
        doc = SourceDocument(content="some text", source="file.pdf")
        assert doc.content == "some text"
        assert doc.source == "file.pdf"
        assert doc.page is None
        assert doc.score is None

    def test_all_fields(self):
        doc = SourceDocument(content="text", source="f.pdf", page=5, score=0.9)
        assert doc.page == 5
        assert doc.score == 0.9


class TestQueryResponse:
    def test_full_response(self):
        resp = QueryResponse(
            answer="Revenue is $42B",
            sources=[SourceDocument(content="...", source="f.pdf")],
            confident=True,
            confidence_score=0.92,
            from_cache=False,
            response_time_ms=123.4,
            timestamp=datetime.now(),
        )
        assert resp.answer == "Revenue is $42B"
        assert resp.confident is True
        assert resp.from_cache is False
        assert len(resp.sources) == 1

    def test_optional_query_rewritten(self):
        resp = QueryResponse(
            answer="test",
            sources=[],
            confident=True,
            confidence_score=1.0,
            from_cache=False,
            response_time_ms=10.0,
            query_rewritten="optimized query",
        )
        assert resp.query_rewritten == "optimized query"


class TestUploadResponse:
    def test_fields(self):
        resp = UploadResponse(
            message="Ingested file.pdf",
            filename="file.pdf",
            chunks_created=47,
            status="success",
        )
        assert resp.chunks_created == 47
        assert resp.status == "success"


class TestChatMessage:
    def test_user_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.confident is None
        assert msg.sources is None

    def test_assistant_message_with_metadata(self):
        msg = ChatMessage(
            role="assistant",
            content="answer",
            confident=True,
            sources=[SourceDocument(content="c", source="s")],
        )
        assert msg.confident is True
        assert len(msg.sources) == 1
