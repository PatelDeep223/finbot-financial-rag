"""Tests for the DB service layer — verifies graceful fallback behavior."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestDbServiceGracefulFallback:
    """Verify that DB service functions handle failures gracefully."""

    def test_save_document_record_failure(self):
        """save_document_record should not raise on DB error."""
        import asyncio
        from app.db import service as db_service

        # Mock session factory to raise
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            # Should not raise
            asyncio.get_event_loop().run_until_complete(
                db_service.save_document_record("test.pdf", 1024, 10)
            )

    def test_list_documents_failure_returns_none(self):
        """list_documents should return None on DB error."""
        import asyncio
        from app.db import service as db_service

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            result = asyncio.get_event_loop().run_until_complete(
                db_service.list_documents()
            )
            assert result is None

    def test_get_conversation_history_failure_returns_none(self):
        """get_conversation_history should return None on DB error."""
        import asyncio
        from app.db import service as db_service

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            result = asyncio.get_event_loop().run_until_complete(
                db_service.get_conversation_history("sess_123")
            )
            assert result is None

    def test_log_query_failure(self):
        """log_query should not raise on DB error."""
        import asyncio
        from app.db import service as db_service

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            asyncio.get_event_loop().run_until_complete(
                db_service.log_query(
                    question="test",
                    rewritten_query=None,
                    answer="answer",
                    confidence_score=0.9,
                    confident=True,
                    from_cache=False,
                    response_time_ms=100.0,
                )
            )

    def test_get_total_query_count_failure_returns_none(self):
        """get_total_query_count should return None on DB error."""
        import asyncio
        from app.db import service as db_service

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            result = asyncio.get_event_loop().run_until_complete(
                db_service.get_total_query_count()
            )
            assert result is None

    def test_delete_conversation_failure(self):
        """delete_conversation_history should not raise on DB error."""
        import asyncio
        from app.db import service as db_service

        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(side_effect=Exception("DB down"))
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            asyncio.get_event_loop().run_until_complete(
                db_service.delete_conversation_history("sess_123")
            )


class TestDbServiceSuccess:
    """Verify DB service functions work correctly with a mocked session."""

    def test_save_document_record_success(self):
        """save_document_record should add a Document and commit."""
        import asyncio
        from app.db import service as db_service

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            asyncio.get_event_loop().run_until_complete(
                db_service.save_document_record("test.pdf", 2048, 15)
            )
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_log_query_success(self):
        """log_query should add a QueryLog and commit."""
        import asyncio
        from app.db import service as db_service

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            asyncio.get_event_loop().run_until_complete(
                db_service.log_query(
                    question="What is revenue?",
                    rewritten_query="Q3 2024 revenue figures",
                    answer="Revenue was $42B",
                    confidence_score=0.92,
                    confident=True,
                    from_cache=False,
                    response_time_ms=1243.5,
                    user_id="user_1",
                    session_id="sess_1",
                )
            )
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_save_conversation_message_success(self):
        """save_conversation_message should add a Conversation and commit."""
        import asyncio
        from app.db import service as db_service

        mock_session = AsyncMock()
        mock_factory = MagicMock()
        mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch.object(db_service, "async_session_factory", mock_factory):
            asyncio.get_event_loop().run_until_complete(
                db_service.save_conversation_message(
                    session_id="sess_1",
                    role="user",
                    content="What is revenue?",
                )
            )
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
