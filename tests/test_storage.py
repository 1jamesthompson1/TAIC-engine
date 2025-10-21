import os
import uuid

import pytest

from app import conversation_store, knowledge_search_store
from backend.Assistant import CompleteHistory


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_conversation_save_and_load(mock_assistant):
    """Test saving and loading conversations."""

    # Create a conversation
    conversation_id = str(uuid.uuid4())
    history = CompleteHistory([])
    history.add_message("user", "Test question")
    history.add_message("assistant", "Test answer")

    username = "testuser"

    # Save conversation
    success = conversation_store.create_or_update_conversation(
        username=username,
        conversation_id=conversation_id,
        history=history,
        conversation_title="Test Conversation",
        db_version=mock_assistant.searcher.db_version,
    )

    assert success

    # Load conversation
    loaded = conversation_store.load_single_conversation(username, conversation_id)
    assert loaded is not None
    assert loaded["conversation_title"] == "Test Conversation"
    assert len(loaded["messages"]) == len(history)

    # Delete conversation
    deleted = conversation_store.delete_conversation(username, conversation_id)
    assert deleted


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_knowledge_search_logging():
    """Test that knowledge searches are logged."""

    searches = knowledge_search_store.get_user_search_history("testuser")
    pre_number = len(searches)

    search_id = str(uuid.uuid4())
    expected_results = 3
    knowledge_search_store.store_search_log(
        username="testuser",
        search_id=search_id,
        search_settings={"query": "test search", "limit": 5},
        results_info={"total_results": expected_results},
        error_info=None,
    )

    # Check that search was logged
    searches = knowledge_search_store.get_user_search_history("testuser")
    assert len(searches) == pre_number + 1

    # Find our search
    our_search = knowledge_search_store.load_detailed_search("testuser", search_id)
    assert our_search is not None
    assert our_search["metadata"]["total_results"] == expected_results

    # Clean up by deleting the search log
    deleted = knowledge_search_store.delete_search_log("testuser", search_id)
    assert deleted
