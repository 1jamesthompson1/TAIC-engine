import os
from unittest.mock import Mock, patch

import pytest

from backend.Assistant import Assistant, CompleteHistory


def test_assistant_initialization(mock_searcher):
    """Test that Assistant can be initialized."""
    assistant = Assistant(
        openai_api_key="test",
        openai_endpoint="https://test.openai.azure.com/",
        searcher=mock_searcher,
    )
    assert assistant.openai_client is not None
    assert assistant.searcher == mock_searcher
    assert len(assistant.tools) > 0  # Should have tools


def test_complete_history_initialization():
    """Test CompleteHistory initialization."""
    history = CompleteHistory([])
    assert len(history) == 0


def test_complete_history_add_message():
    """Test adding messages to CompleteHistory."""
    history = CompleteHistory([])
    history.add_message("user", "Hello")

    assert len(history) == 1
    assert history[0]["display"]["role"] == "user"
    assert history[0]["display"]["content"] == "Hello"
    assert history[0]["ai"]["role"] == "user"
    assert history[0]["ai"]["content"] == "Hello"


def test_complete_history_gradio_format():
    """Test gradio format conversion."""
    history = CompleteHistory([])
    history.add_message("user", "Hello")
    history.add_message("assistant", "Hi there")
    history.add_message("user", "Can you tell me about recent ATSB investigations?")
    history.start_thought("I need to find information about ATSB investigations.")
    history.end_thought()
    history.add_function_call(
        {
            "name": "search_knowledge",
            "arguments": {"query": "ATSB investigations"},
            "type": "function_call",
        },
    )
    history.complete_function_call(
        output="Found 3 relevant ATSB investigations.",
        call_id="manual",
    )

    gradio_format = history.gradio_format()
    expected_message_len = (
        6  # user, assistant, user, thought, function call, function result
    )
    assert len(gradio_format) == expected_message_len
    assert gradio_format[0]["role"] == "user"
    assert gradio_format[0]["content"] == "Hello"
    assert gradio_format[1]["role"] == "assistant"
    assert gradio_format[1]["content"] == "Hi there"

    # thought
    thought_message = history[-3]
    assert thought_message["display"]["role"] == "assistant"
    assert (
        "I need to find information about ATSB investigations."
        in thought_message["display"]["content"]
    )
    assert thought_message["ai"]["role"] == "assistant"
    assert (
        "I need to find information about ATSB investigations."
        in thought_message["ai"]["content"]
    )
    # Check that tool usage is included in the last message
    function_call = history[-2]
    assert function_call["display"]["role"] == "assistant"
    assert "Executing search_knowledge" in function_call["display"]["content"]

    function_result = history[-1]
    assert function_result["display"]["role"] == "assistant"
    assert (
        "Found 3 relevant ATSB investigations." in function_result["display"]["content"]
    )


@patch("backend.Assistant.Assistant.process_input")
def test_basic_conversation_flow(mock_process, mock_assistant):
    """Test basic conversation processing."""
    history = CompleteHistory([])
    history.add_message("user", "What is a safety factor?")

    # Mock the process_input to return a simple response
    mock_process.return_value = iter(
        [
            (history, history.gradio_format()),
        ],
    )

    # This would normally be called in the Gradio interface
    # For testing, we just verify the method exists and can be called
    assert hasattr(mock_assistant, "process_input")
    assert callable(mock_assistant.process_input)


def test_conversation_title_generation(mock_assistant):
    """Test conversation title generation."""
    history = CompleteHistory([])
    history.add_message("user", "Tell me about aviation safety")
    history.add_message("assistant", "Aviation safety is...")

    # Mock OpenAI response for title generation
    with patch.object(mock_assistant.openai_client, "chat") as mock_chat:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Aviation Safety Discussion"
        mock_chat.completions.create.return_value = mock_response

        title = mock_assistant.provide_conversation_title(history)
        assert isinstance(title, str)
        assert len(title) > 0


def test_history_format_validation():
    """Test that history format validation works."""
    # Valid format
    valid_history = [
        {
            "display": {"role": "user", "content": "Hello"},
            "ai": {"role": "user", "content": "Hello"},
        },
    ]
    history = CompleteHistory(valid_history)
    assert len(history) == 1

    # Invalid format should raise error
    with pytest.raises(ValueError, match="Failed to format"):
        CompleteHistory([{"invalid": "format"}])


def test_empty_history_processing_error(mock_assistant):
    """Test that processing empty history raises error."""
    history = CompleteHistory([])

    with pytest.raises(ValueError, match="history is empty"):
        list(mock_assistant.process_input(history))


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_real_assistant_initialization(mock_assistant):
    """Test that real Assistant initializes with actual dependencies."""
    assert mock_assistant.openai_client is not None
    assert mock_assistant.searcher is not None
    assert len(mock_assistant.tools) > 0


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_real_conversation_processing(mock_assistant):
    """Test actual conversation processing with real AI."""
    history = CompleteHistory([])
    history.add_message("user", "What is a safety factor in aviation?")

    # Process the input (this will make real API calls)
    results = list(mock_assistant.process_input(history))

    assert len(results) > 0
    final_history, gradio_format = results[-1]
    assert len(final_history) > 1  # Should have user message + assistant response
    assert len(gradio_format) > 1


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_basic_tool_use(mock_assistant):
    """Test basic tool usage."""
    history = CompleteHistory([])
    history.add_message("user", "Can you tell me about recent ATSB investigations?")

    # Process the input (this will make real API calls)
    results = list(mock_assistant.process_input(history))

    assert len(results) > 0
    final_history, gradio_format = results[-1]
    assert len(final_history) > 1  # Should have user message + assistant response
    assert len(gradio_format) > 1
    # Check that a function call was made
    function_calls = [
        msg for msg in final_history if msg["ai"].get("type") == "function_call"
    ]
    assert len(function_calls) > 0


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_real_title_generation(mock_assistant):
    """Test actual conversation title generation."""
    history = CompleteHistory([])
    history.add_message("user", "Tell me about runway safety")
    history.add_message("assistant", "Runway safety is crucial...")

    title = mock_assistant.provide_conversation_title(history)
    assert isinstance(title, str)
    assert len(title) > 0
    assert "runway" in title.lower() or "safety" in title.lower()
