import io
import pytest


class TestUploadEndpoint:
    def test_upload_txt(self, client, mock_ingest, sample_txt):
        with open(sample_txt, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("report.txt", f, "text/plain")}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "report.txt"
        assert data["chunks_created"] == 10
        assert data["status"] == "success"
        assert "successfully" in data["message"].lower()

    def test_upload_pdf(self, client, mock_ingest, sample_pdf):
        with open(sample_pdf, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"file": ("earnings.pdf", f, "application/pdf")}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "earnings.pdf"
        assert data["chunks_created"] == 10
        assert data["status"] == "success"

    def test_upload_invalid_extension(self, client):
        fake_csv = io.BytesIO(b"col1,col2\n1,2")
        response = client.post(
            "/api/v1/upload",
            files={"file": ("data.csv", fake_csv, "text/csv")}
        )
        assert response.status_code == 400
        assert "pdf" in response.json()["detail"].lower() or "txt" in response.json()["detail"].lower()

    def test_upload_invalid_extension_xlsx(self, client):
        fake = io.BytesIO(b"fake excel")
        response = client.post(
            "/api/v1/upload",
            files={"file": ("data.xlsx", fake, "application/vnd.openxmlformats")}
        )
        assert response.status_code == 400

    def test_upload_no_file(self, client):
        response = client.post("/api/v1/upload")
        assert response.status_code == 422  # missing required field
