class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        response = client.get("/api/v1/stats")
        assert response.status_code == 200

    def test_stats_response_structure(self, client):
        data = client.get("/api/v1/stats").json()
        assert "total_queries" in data
        assert "cache_hit_rate" in data
        assert "cache_stats" in data
        assert "vectorstore_loaded" in data
        assert "uptime_seconds" in data

    def test_stats_initial_values(self, client):
        data = client.get("/api/v1/stats").json()
        assert data["total_queries"] == 0
        assert data["cache_hit_rate"] == 0

    def test_stats_after_queries(self, client, mock_pipeline_query):
        # Make 2 queries
        client.post("/api/v1/query", json={"question": "test 1"})
        client.post("/api/v1/query", json={"question": "test 2"})

        data = client.get("/api/v1/stats").json()
        assert data["total_queries"] == 2

    def test_stats_cache_hit_rate(self, client, mock_pipeline_query_cached):
        client.post("/api/v1/query", json={"question": "test"})

        data = client.get("/api/v1/stats").json()
        assert data["total_queries"] == 1
        assert data["cache_hit_rate"] == 100.0
