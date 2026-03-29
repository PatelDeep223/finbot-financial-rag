class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["service"] == "FinBot RAG API"
        assert data["version"] == "1.0.0"
