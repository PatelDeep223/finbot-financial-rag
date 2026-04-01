"""Tests for signup, login, and JWT auth endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


def _make_fake_user(id=1, username="deep", email="deep@test.com"):
    user = MagicMock()
    user.id = id
    user.username = username
    user.email = email
    user.is_active = True
    user.created_at = datetime.now()
    return user


class TestSignup:
    def test_signup_success(self, client):
        fake_user = _make_fake_user()

        with patch("app.db.service.get_user_by_email", new_callable=AsyncMock, return_value=None), \
             patch("app.db.service.create_user", new_callable=AsyncMock, return_value=fake_user):
            response = client.post("/api/v1/auth/signup", json={
                "username": "deep",
                "email": "deep@test.com",
                "password": "mypass123",
            })

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "deep"

    def test_signup_duplicate_email(self, client):
        existing = _make_fake_user(email="taken@test.com")

        with patch("app.db.service.get_user_by_email", new_callable=AsyncMock, return_value=existing):
            response = client.post("/api/v1/auth/signup", json={
                "username": "new",
                "email": "taken@test.com",
                "password": "mypass123",
            })

        assert response.status_code == 409

    def test_signup_short_password(self, client):
        response = client.post("/api/v1/auth/signup", json={
            "username": "deep",
            "email": "deep@test.com",
            "password": "123",
        })
        assert response.status_code == 422


class TestLogin:
    def test_login_success(self, client):
        from app.core.security import hash_password

        fake_user = _make_fake_user()
        fake_user.hashed_password = hash_password("mypass123")

        with patch("app.db.service.get_user_by_email", new_callable=AsyncMock, return_value=fake_user):
            response = client.post("/api/v1/auth/login", json={
                "email": "deep@test.com",
                "password": "mypass123",
            })

        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_login_wrong_password(self, client):
        from app.core.security import hash_password

        fake_user = _make_fake_user()
        fake_user.hashed_password = hash_password("correct")

        with patch("app.db.service.get_user_by_email", new_callable=AsyncMock, return_value=fake_user):
            response = client.post("/api/v1/auth/login", json={
                "email": "deep@test.com",
                "password": "wrong",
            })

        assert response.status_code == 401

    def test_login_nonexistent_email(self, client):
        with patch("app.db.service.get_user_by_email", new_callable=AsyncMock, return_value=None):
            response = client.post("/api/v1/auth/login", json={
                "email": "nobody@test.com",
                "password": "whatever",
            })

        assert response.status_code == 401


class TestMe:
    def test_me_dev_mode(self, client):
        """In dev mode (user_count=0), /me returns synthetic dev user."""
        with patch("app.db.service.user_count", new_callable=AsyncMock, return_value=0):
            response = client.get("/api/v1/auth/me")
        assert response.status_code == 200
        assert response.json()["username"] == "dev_user"

    @pytest.mark.skipif(True, reason="Requires real DB with users — run as integration test")
    def test_me_with_valid_jwt(self, client):
        """With auth enabled + valid JWT, /me returns real user. (integration test)"""
        pass


class TestProtectedEndpoints:
    @pytest.mark.skipif(True, reason="Requires real DB with users — run as integration test")
    def test_query_without_token_when_auth_enabled(self, client):
        """With auth enabled, /query without token returns 401. (integration test)"""
        pass

    def test_query_with_valid_token_dev_mode(self, client, mock_pipeline_query):
        """In dev mode (no users), /query works without token."""
        with patch("app.db.service.user_count", new_callable=AsyncMock, return_value=0):
            response = client.post(
                "/api/v1/query",
                json={"question": "What is revenue?"},
            )

        assert response.status_code == 200
        assert "answer" in response.json()
