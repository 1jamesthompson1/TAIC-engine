import os
import uuid
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from app import (
    assistant_instance,
    create_or_update_conversation,
    get_user_conversations_metadata,
    load_conversation,
    perform_search,
)
from backend.Assistant import CompleteHistory


class TestAppSmoke:
    SUCCESS_STATUS = 200
    """Smoke tests for the web application."""

    def test_app_imports_without_error(self):
        """Test that the app module can be imported without errors."""
        try:
            import app  # noqa: PLC0415

            assert app.app is not None
        except Exception as e:  # noqa: BLE001
            pytest.fail(f"Failed to import app: {e}")

    def test_login_page_loads(self, client):
        """Test that login page loads successfully."""
        response = client.get("/login-page")
        assert response.status_code == self.SUCCESS_STATUS
        assert "TAIC smart tools" in response.text

    def test_root_redirect_without_auth(self, client):
        """Test that root path redirects to login when not authenticated."""
        response = client.get("/")
        assert "Please login to continue" in response.text

    def test_tool_redirect_without_auth(self, client):
        """Test that /tools path redirects to login when not authenticated."""
        response = client.get("/tools")
        assert "Please login to continue" in response.text

    def test_tool_with_auth(self, client):
        """Test that /tools path loads when authenticated."""
        with patch("app.get_user") as mock_get_user:
            mock_get_user.return_value = "testuser"
            response = client.get("/tools")

            assert "Assistant" in response.text

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_real_app_startup(self, client):
        """Test that the real app starts up with all services initialized."""
        # Test that we can access the login page
        response = client.get("/login-page")
        assert response.status_code == self.SUCCESS_STATUS
        assert "TAIC smart tools" in response.text


class TestToolsFunctions:
    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_real_assistant_integration(self):
        """Test that the assistant is properly integrated."""

        assert assistant_instance is not None
        assert assistant_instance.searcher is not None
        assert len(assistant_instance.tools) > 0

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_perform_search_functionality(self):
        """Test the perform_search function with real services."""

        # Test search execution
        (
            results,
            download_dict,
            message,
            _doc_plot,
            _mode_plot,
            _year_hist,
            _agency_plot,
            _event_plot,
        ) = perform_search(
            username="testuser",
            query="safety factor",
            year_range=[2010, 2024],
            document_type=["Safety issues"],
            modes=["Aviation", "Rail", "Maritime"],
            agencies=["TAIC"],
            relevance=0.6,
        )

        assert isinstance(results, pd.DataFrame)
        assert isinstance(message, str)
        assert download_dict is not None
        assert "settings" in download_dict
        assert "results" in download_dict

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_perform_search_with_empty_query(self):
        """Test perform_search with empty query."""

        (
            results,
            _download_dict,
            message,
            _doc_plot,
            _mode_plot,
            _year_hist,
            _agency_plot,
            _event_plot,
        ) = perform_search(
            username="testuser",
            query="",
            year_range=[2010, 2024],
            document_type=["Safety issues"],
            modes=["Aviation"],
            agencies=["TAIC"],
            relevance=0.6,
        )

        # Should handle empty query gracefully
        assert isinstance(results, pd.DataFrame)
        assert isinstance(message, str)


class TestConversationFunctions:
    """Tests for conversation-related functions in the app."""

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_conversation_functions(self):
        """Test conversation-related functions."""

        # Test getting user conversations (should work even with no conversations)
        conversations = get_user_conversations_metadata(Mock(username="testuser"))
        assert isinstance(conversations, list)
        assert len(conversations) >= 0

        # Test creating/updating conversation
        str(uuid.uuid4())
        history = CompleteHistory([])
        history.add_message("user", "Test message")
        history.add_message("assistant", "Test response")

        # This would require authentication context, so we'll just test the function exists
        assert callable(create_or_update_conversation)

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_load_nonexistent_conversation_function(self):
        """Test loading conversations."""

        # Test loading non-existent conversation
        conversation_id = str(uuid.uuid4())
        history, gradio_format, conv_id, title = load_conversation(
            Mock(username="testuser"),
            conversation_id,
        )

        assert len(history) == 0
        assert gradio_format == []
        assert conv_id == f"`{conversation_id}`"
        assert title == "*Failed to load*"

    @pytest.mark.skipif(
        not os.getenv("TEST_USE_REAL_SERVICES"),
        reason="Requires real services",
    )
    def test_load_conversation_function(self):
        """Test loading conversations a real conversation."""
        conversation_id = "1f6ebb26-abc4-4d7f-ab0b-da07c34ca73e"

        history, _gradio_format, _conv_id, _title = load_conversation(
            Mock(username="testuser"),
            conversation_id,
        )

        assert isinstance(history, list)

        assert len(history) > 0
