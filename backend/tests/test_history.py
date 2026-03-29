class TestHistoryEndpoints:
    def test_get_history_empty(self, client):
        response = client.get("/api/v1/history/unknown_session")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "unknown_session"
        assert data["messages"] == []

    def test_get_history_after_query(self, client, mock_pipeline_query):
        # Make a query with a session_id
        client.post("/api/v1/query", json={
            "question": "What is revenue?",
            "session_id": "sess_1"
        })

        response = client.get("/api/v1/history/sess_1")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess_1"
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "What is revenue?"
        assert data["messages"][1]["role"] == "assistant"

    def test_get_history_multiple_queries(self, client, mock_pipeline_query):
        client.post("/api/v1/query", json={"question": "Q1", "session_id": "sess_2"})
        client.post("/api/v1/query", json={"question": "Q2", "session_id": "sess_2"})

        data = client.get("/api/v1/history/sess_2").json()
        assert len(data["messages"]) == 4  # 2 user + 2 assistant

    def test_clear_history(self, client, mock_pipeline_query):
        # Add some history
        client.post("/api/v1/query", json={"question": "test", "session_id": "sess_3"})

        # Verify it exists
        data = client.get("/api/v1/history/sess_3").json()
        assert len(data["messages"]) == 2

        # Clear it
        response = client.delete("/api/v1/history/sess_3")
        assert response.status_code == 200
        assert "cleared" in response.json()["message"].lower()

        # Verify it's gone
        data = client.get("/api/v1/history/sess_3").json()
        assert data["messages"] == []

    def test_clear_nonexistent_history(self, client):
        """Clearing a session that doesn't exist should still succeed."""
        response = client.delete("/api/v1/history/nonexistent")
        assert response.status_code == 200
