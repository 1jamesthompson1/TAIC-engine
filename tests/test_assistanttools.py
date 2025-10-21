import os

import pytest

from backend.AssistantTools import ReadReportTool, SearchTool


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_tool_execution_search(mock_searcher):
    """Test that search tool actually executes."""

    search_tool = SearchTool(mock_searcher)
    result = search_tool.execute(
        query="safety factor",
        document_type=["safety_issue"],
        year_range=(2010, 2024),
        modes=["0", "1", "2"],
        agencies=["TAIC"],
        search_type="vector",
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert "table" in result.lower()  # Should return a html table


@pytest.mark.skipif(
    not os.getenv("TEST_USE_REAL_SERVICES"),
    reason="Requires real services",
)
def test_tool_execution_read_report(mock_searcher):
    """Test that read report tool actually executes."""

    read_tool = ReadReportTool(mock_searcher)

    # Use a known report ID from the mock or real database
    known_report_id = "ATSB_m_2021_003"
    result = read_tool.execute(report_id=known_report_id)

    assert "Not yet implemented" in result  # Since it's a placeholder
