import pytest


class TestQueryEndpoint:
    def test_query_success(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "What is the Q3 revenue?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert "confident" in data
        assert "confidence_score" in data
        assert "from_cache" in data
        assert "response_time_ms" in data
        assert "timestamp" in data
        assert data["from_cache"] is False

    def test_query_with_sources(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "What is revenue?"
        })
        data = response.json()
        assert len(data["sources"]) > 0
        source = data["sources"][0]
        assert "content" in source
        assert "source" in source
        assert source["source"] == "earnings.pdf"

    def test_query_empty_question(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "   "
        })
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_query_too_long(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "x" * 1001
        })
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()

    def test_query_exactly_1000_chars(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "x" * 1000
        })
        assert response.status_code == 200

    def test_query_missing_question_field(self, client):
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422  # Pydantic validation error

    def test_query_default_user_id(self, client, mock_pipeline_query):
        """user_id defaults to 'anonymous' when not provided."""
        response = client.post("/api/v1/query", json={
            "question": "test"
        })
        assert response.status_code == 200

    def test_query_with_user_id(self, client, mock_pipeline_query):
        response = client.post("/api/v1/query", json={
            "question": "test",
            "user_id": "user_123"
        })
        assert response.status_code == 200

    def test_query_cached_response(self, client, mock_pipeline_query_cached):
        response = client.post("/api/v1/query", json={
            "question": "What is the Q3 revenue?"
        })
        data = response.json()
        assert data["from_cache"] is True
        assert data["response_time_ms"] < 100  # cached should be fast

    def test_query_demo_mode(self, client, mock_pipeline_demo):
        """When no documents are loaded, returns demo response."""
        response = client.post("/api/v1/query", json={
            "question": "What is revenue?"
        })
        data = response.json()
        assert response.status_code == 200
        assert "upload" in data["answer"].lower() or "no documents" in data["answer"].lower()
        assert data["sources"] == []
        assert data["confident"] is True

    def test_query_with_session_stores_history(self, client, mock_pipeline_query):
        from tests.conftest import pipeline

        response = client.post("/api/v1/query", json={
            "question": "What is revenue?",
            "session_id": "sess_abc"
        })
        assert response.status_code == 200
        assert "sess_abc" in pipeline.conversation_history
        assert len(pipeline.conversation_history["sess_abc"]) == 2  # user + assistant
