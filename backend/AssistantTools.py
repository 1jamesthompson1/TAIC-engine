from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from .Searching import Searcher, SearchParams


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the tool does."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON schema for the tool parameters."""

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters and return result as string."""

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class SearchTool(Tool):
    """Tool for searching the knowledge base."""

    def __init__(self, searcher: Searcher):
        self.searcher = searcher

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return """Search for safety issues and recommendations from the New Zealand Transport Accident Investigation Commission. This function searches a vector database.
        Example usage:
        search(query="What are the common causes of aviation accidents?", type="vector", year_range=[2010, 2023], document_type=["safety_issue", "recommendation"], modes=["0"], agencies=["TAIC"])

        search(query="What safety issues are associated with runway incursions?", type="vector", year_range=[2000, 2023], document_type=["safety_issue", "recommendations"], modes=["0"], agencies=["TAIC", "ATSB"])

        search(query="What are some recent accidents?", type="vector", year_range=[2000, 2023], document_type=["summary"], modes=["0", "1", "2"], agencies=["ATSB", "TSB", "TAIC"])
    """

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for. If left as an empty string it will return all results that match the other parameters.",
                },
                "search_type": {
                    "type": "string",
                    "enum": ["fts", "vector"],
                    "description": "The type of search to perform. fts should be used if the query is asking a specific question about an organisation, etc. Otherwise for more general information use vector, it will embed your query and search the vector database.",
                },
                "year_range": {
                    "type": "array",
                    "description": f"An array specifying the start and end years for filtering results. Valid range is 2000-{datetime.now(tz=timezone.utc).year}.",
                    "items": {"type": "number"},
                },
                "document_type": {
                    "type": "array",
                    "description": "A list of document types to filter the search results. Safety issues and recommendations follow definitions given, while report sections are reports chunked into sections/pages, summary are brief overviews of the reports scrapped from the agencies report webpages only available for TAIC and ATSB. Valid types are 'safety_issue', 'recommendation', 'section' and 'summary'.",
                    "items": {"type": "string"},
                },
                "modes": {
                    "type": "array",
                    "description": "A list of modes to filter the search results. Valid modes are 0, 1, and 2. Which are aviation, rail, and marine respectively.",
                    "items": {"type": "string"},
                },
                "agencies": {
                    "type": "array",
                    "description": "A list of agencies to filter the search results. Valid agencies are TSB, ATSB, and TAIC. These are Transport Safety Board (Canada), Australian Transport Safety Board, and Transport Accident Investigation Commission (New Zealand) respectively.",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "query",
                "search_type",
                "year_range",
                "document_type",
                "modes",
                "agencies",
            ],
        }

    def execute(self, **kwargs) -> str:
        results, info, _plots = self.searcher.knowledge_search(
            SearchParams(**kwargs),
        )

        results_html = results.to_html(index=False)

        return f"<p>Information about the search:<br>{info}<br>Search Results:<br></p>{results_html}"


class ReadReportTool(Tool):
    """Tool for reading individual reports."""

    def __init__(self, searcher):
        self.searcher = searcher

    @property
    def name(self) -> str:
        return "read_report"

    @property
    def description(self) -> str:
        return "Read the full content of a specific report by its ID."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "report_id": {
                    "type": "string",
                    "description": "The ID of the report to read.",
                },
            },
            "required": ["report_id"],
        }

    def execute(self, **_kwargs) -> str:
        return "Not yet implemented"


class ReasoningTool(Tool):
    """Tool for sending information to a powerful reasoning model."""

    def __init__(self, openai_client, reasoning_model: str = "gpt-4.1"):
        self.openai_client = openai_client
        self.reasoning_model = reasoning_model

    @property
    def name(self) -> str:
        return "reason"

    @property
    def description(self) -> str:
        return "Send information to a powerful reasoning model for deeper analysis."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "information": {
                    "type": "string",
                    "description": "The information to analyze.",
                },
                "task": {
                    "type": "string",
                    "description": "The reasoning task to perform.",
                },
            },
            "required": ["information", "task"],
        }

    def execute(self, **kwargs) -> str:
        information = kwargs.get("information", "")
        task = kwargs.get("task", "")

        messages = [
            {
                "role": "system",
                "content": "You are a powerful reasoning AI. Analyze the provided information and perform the specified task.",
            },
            {"role": "user", "content": f"Task: {task}\n\nInformation: {information}"},
        ]
        response = self.openai_client.responses.create(
            model=self.reasoning_model,
            input=messages,
        )
        return response.output_text
