import os
from tests.conftest import mock_settings


class TestDocumentsEndpoint:
    def test_list_documents_empty(self, client):
        # Point to an empty temp dir (already set by conftest mock_settings)
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_list_documents_after_file_placed(self, client, tmp_path, monkeypatch):
        """If there are files in the documents dir, they appear in the list."""
        # Create a fake document in a temp dir
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "annual_report.pdf").write_bytes(b"fake pdf content")

        monkeypatch.setattr(mock_settings, "DOCUMENTS_PATH", str(docs_dir))

        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "annual_report.pdf"
        assert "size_kb" in data["documents"][0]
        assert "modified" in data["documents"][0]
