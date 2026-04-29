"""OpenAI API caller with cost tracking and model management.

This module provides an interface to interact with OpenAI and Azure OpenAI APIs,
including cost tracking, token management, and support for structured outputs.
"""

import os
import warnings
from dataclasses import dataclass
from threading import Lock

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

load_dotenv()


# Custom Exceptions
class ClientNotAvailableError(Exception):
    """Raised when OpenAI/Azure client is not configured."""

    def __init__(self):
        """Initialize ClientNotAvailableError."""
        super().__init__(
            "OpenAI client is not available. Please set either OPENAI_API_KEY or both "
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
        )


class ModelPricingNotDefinedError(Exception):
    """Raised when pricing for a model is not defined."""

    def __init__(self, model: str):
        """Initialize ModelPricingNotDefinedError.

        Args:
            model: The model name for which pricing is not defined.
        """
        super().__init__(f"Pricing for model {model} is not defined.")


class ModelNotFoundError(Exception):
    """Raised when a specified model is not available."""

    def __init__(self, model: str, available_models: list[str]):
        """Initialize ModelNotFoundError.

        Args:
            model: The model name that was not found.
            available_models: List of available model names.
        """
        super().__init__(
            f"Model {model} not found. Available models: {available_models}"
        )


class AICallerFailedError(Exception):
    """Raised when an AI call fails due to an API error."""

    def __init__(self, message: str):
        """Initialize AICallerFailedError.

        Args:
            message: The error message describing the failure.
        """
        super().__init__(f"AI call failed: {message}")


class AIRefusalError(Exception):
    """Raised when the AI model refuses to respond to a query."""

    def __init__(self, message: str):
        """Initialize AIRefusalError.

        Args:
            message: The error message describing the refusal.
        """
        super().__init__(f"Model refused to respond: {message}")


class QueryTooLongError(Exception):
    """Raised when a query exceeds the token limit."""

    def __init__(self, length: int, limit: int):
        """Initialize QueryTooLongError.

        Args:
            length: The actual length of the query in tokens.
            limit: The maximum allowed token limit.
        """
        super().__init__(
            f"The combined system and user query exceeds the token limit. Length: {length}, Limit: {limit}."
        )


# Initialize clients only if API keys are available
openai_client = None

# Cost tracking (thread-safe for CI environments)
_cost_lock = Lock()
_api_costs = {
    "total_cost": 0.0,
    "input_cost": 0.0,
    "cached_input_cost": 0.0,
    "output_cost": 0.0,
    "total_input_tokens": 0,
    "total_cached_tokens": 0,
    "total_output_tokens": 0,
    "calls": 0,
    "by_model": {},
}


@dataclass(frozen=True)
class ModelDefinition:
    """Definition for a model managed by AICaller."""

    api_model: str
    limit: int
    pricing: dict[str, float]


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    "gpt-4": ModelDefinition(
        api_model="gpt-4o",
        limit=128_000,
        pricing={"input": 5.00, "cached_input": 0.50, "output": 15.00},
    ),
    "gpt-5-mini": ModelDefinition(
        api_model="gpt-5-mini",
        limit=400_000,
        pricing={"input": 0.25, "cached_input": 0.025, "output": 2.00},
    ),
    # Intermittently has problems with not generating as blocked by "I'm sorry I can't respond to that..." error. Otherwise cheap and high reasoning.
    "gpt-5.4-mini": ModelDefinition(
        api_model="gpt-5.4-mini",
        limit=400_000,
        pricing={"input": 0.75, "cached_input": 0.075, "output": 4.50},
    ),
    "gpt-5.4-nano": ModelDefinition(
        api_model="gpt-5.4-nano",
        limit=400_000,
        pricing={"input": 0.2, "cached_input": 0.02, "output": 1.25},
    ),
    # Works well yet costs more. Limited reasoning
    "gpt-5.1-chat": ModelDefinition(
        api_model="gpt-5.1-chat",
        limit=400_000,
        pricing={"input": 1.25, "cached_input": 0.125, "output": 10.00},
    ),
}

# Pricing per 1M tokens in USD
_PRICING = {
    definition.api_model: definition.pricing for definition in MODEL_REGISTRY.values()
}


# For OpenAI, prioritize Azure OpenAI if credentials are available, otherwise use standard OpenAI
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")

if azure_api_key and azure_endpoint:
    # Use Azure OpenAI
    openai_client = OpenAI(api_key=azure_api_key, base_url=azure_endpoint)
elif openai_api_key:
    # Use standard OpenAI
    openai_client = OpenAI(api_key=openai_api_key)


def track_api_cost(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
):
    """Track API costs for monitoring purposes.

    Args:
        model: The model name for which to track costs.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        cached_tokens: Number of cached input tokens, defaults to 0.

    Raises:
        ModelPricingNotDefinedError: If pricing for the model is not defined.
    """
    if model not in _PRICING:
        raise ModelPricingNotDefinedError(model)

    pricing = _PRICING[model]
    regular_input_tokens = input_tokens - cached_tokens
    input_cost = regular_input_tokens * pricing["input"] / 1_000_000
    cached_input_cost = cached_tokens * pricing["cached_input"] / 1_000_000
    output_cost = output_tokens * pricing["output"] / 1_000_000
    total_cost = input_cost + cached_input_cost + output_cost

    with _cost_lock:
        _api_costs["total_cost"] += total_cost
        _api_costs["input_cost"] += input_cost
        _api_costs["cached_input_cost"] += cached_input_cost
        _api_costs["output_cost"] += output_cost
        _api_costs["total_input_tokens"] += input_tokens
        _api_costs["total_cached_tokens"] += cached_tokens
        _api_costs["total_output_tokens"] += output_tokens
        _api_costs["calls"] += 1

        if model not in _api_costs["by_model"]:
            _api_costs["by_model"][model] = {
                "total_cost": 0.0,
                "input_cost": 0.0,
                "cached_input_cost": 0.0,
                "output_cost": 0.0,
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "calls": 0,
            }

        _api_costs["by_model"][model]["total_cost"] += total_cost
        _api_costs["by_model"][model]["input_cost"] += input_cost
        _api_costs["by_model"][model]["cached_input_cost"] += cached_input_cost
        _api_costs["by_model"][model]["output_cost"] += output_cost
        _api_costs["by_model"][model]["input_tokens"] += input_tokens
        _api_costs["by_model"][model]["cached_tokens"] += cached_tokens
        _api_costs["by_model"][model]["output_tokens"] += output_tokens
        _api_costs["by_model"][model]["calls"] += 1


def get_api_costs():
    """Get current API cost tracking data.

    Returns:
        dict: Dictionary containing cost breakdown by model and totals.
    """
    with _cost_lock:
        return {
            "total_cost": round(_api_costs["total_cost"], 4),
            "input_cost": round(_api_costs["input_cost"], 4),
            "cached_input_cost": round(_api_costs["cached_input_cost"], 4),
            "output_cost": round(_api_costs["output_cost"], 4),
            "total_input_tokens": _api_costs["total_input_tokens"],
            "total_cached_tokens": _api_costs["total_cached_tokens"],
            "total_output_tokens": _api_costs["total_output_tokens"],
            "calls": _api_costs["calls"],
            "by_model": {
                k: {
                    "total_cost": round(v["total_cost"], 4),
                    "input_cost": round(v["input_cost"], 4),
                    "cached_input_cost": round(v["cached_input_cost"], 4),
                    "output_cost": round(v["output_cost"], 4),
                    "input_tokens": v["input_tokens"],
                    "cached_tokens": v["cached_tokens"],
                    "output_tokens": v["output_tokens"],
                    "calls": v["calls"],
                }
                for k, v in _api_costs["by_model"].items()
            },
        }


def print_api_cost_summary(costs_data=None):
    """Print API cost summary to stdout (used by pytest hook)."""
    costs = costs_data if costs_data is not None else get_api_costs()

    if costs["calls"] > 0:
        console = Console()

        table = Table(title="API COST SUMMARY", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", min_width=12, no_wrap=False)
        table.add_column("Calls", justify="right")
        table.add_column("In Tok", justify="right")
        table.add_column("Cac", justify="right")
        table.add_column("Out", justify="right")
        table.add_column("In$", justify="right", style="green")
        table.add_column("Cac$", justify="right", style="green")
        table.add_column("Out$", justify="right", style="green")
        table.add_column("Tot$", justify="right", style="yellow bold")

        # Per-model rows
        for model, data in costs["by_model"].items():
            table.add_row(
                model,
                str(data["calls"]),
                f"{data['input_tokens']:,}",
                f"{data['cached_tokens']:,}",
                f"{data['output_tokens']:,}",
                f"${data['input_cost']:.3f}",
                f"${data['cached_input_cost']:.3f}",
                f"${data['output_cost']:.3f}",
                f"${data['total_cost']:.3f}",
            )

        if len(costs["by_model"]) > 1:
            # Total row
            table.add_row(
                "[bold]TOTAL[/bold]",
                str(costs["calls"]),
                f"{costs['total_input_tokens']:,}",
                f"{costs['total_cached_tokens']:,}",
                f"{costs['total_output_tokens']:,}",
                f"${costs['input_cost']:.3f}",
                f"${costs['cached_input_cost']:.3f}",
                f"${costs['output_cost']:.3f}",
                f"[bold yellow]${costs['total_cost']:.3f}[/bold yellow]",
            )

        console.print("\n")
        console.print(table)
        console.print("")


def reset_api_costs():
    """Reset cost tracking (useful for test isolation).

    This function resets all accumulated API cost data to zero.
    """
    with _cost_lock:
        _api_costs = {
            "total_cost": 0.0,
            "input_cost": 0.0,
            "cached_input_cost": 0.0,
            "output_cost": 0.0,
            "total_input_tokens": 0,
            "total_cached_tokens": 0,
            "total_output_tokens": 0,
            "calls": 0,
            "by_model": {},
        }


class BaseAICaller:
    """Base class for AI API callers.

    This class provides common functionality for interacting with AI models,
    including token counting and query validation.
    """

    def __init__(self, client, model, limit):
        """Initialize the BaseAICaller.

        Args:
            client: The OpenAI/Azure client instance.
            model: The model name/identifier.
            limit: Maximum token limit for queries.
        """
        self.client = client
        self.model = model
        self.limit = limit

    @staticmethod
    def get_tokens(texts):
        """Count tokens in the given texts.

        Args:
            texts: List of text strings to count tokens for.

        Returns:
            list: List of token counts corresponding to each text.
        """
        # No need to check for model as all models have the same or broadly similar tokenization
        enc = tiktoken.encoding_for_model("gpt-5")
        return [len(enc.encode(text)) for text in texts]

    def check_query_above_limit(self, query):
        """Check if a query exceeds the token limit.

        Args:
            query: The query string to check.

        Raises:
            QueryTooLongError: If the query exceeds the token limit.
        """
        length = sum(self.get_tokens([query]))
        if length > self.limit:
            raise QueryTooLongError(length, self.limit)


class OpenAICaller(BaseAICaller):
    """OpenAI/Azure OpenAI API caller.

    Handles queries to OpenAI or Azure OpenAI models with support for structured
    outputs, reasoning, and cost tracking.
    """

    def __init__(self, client, model, limit):
        """Initialize the OpenAICaller.

        Args:
            client: The OpenAI/Azure client instance.
            model: The model name/identifier.
            limit: Maximum token limit for queries.
        """
        super().__init__(client, model, limit)

    def query(  # noqa: PLR0912, PLR0913, PLR0917
        self,
        system,
        user,
        temp,
        max_tokens=32_000,
        output_structure=None,
        reasoning=None,
        raw_output=False,
    ):
        """Query the OpenAI/Azure OpenAI model.

        Args:
            system: System prompt/instructions for the model.
            user: User message/query.
            temp: Temperature parameter for response generation.
            max_tokens: Maximum tokens in the response (default: 32000).
            output_structure: Optional structured output format.
            reasoning: Reasoning effort level if applicable.
            raw_output: If True, return raw response object, else formatted output.

        Returns:
            The model's response (formatted or raw based on raw_output parameter),
            or None if an error occurs or query exceeds limits.

        Raises:
            ClientNotAvailableError: If OpenAI/Azure client is not configured.
            AICallerFailedError: If the API call fails due to an error.
            ValidationError: If parsing the model response into the requested
                structured format fails (JSON validation error).
            AIRefusalError: If the model appears to refuse to answer (detected
                when validation error message contains "I'm sorry").
        """
        if self.client is None:
            raise ClientNotAvailableError()

        self.check_query_above_limit(system + user)

        if (
            reasoning != "none"
            and temp is not None
            and not self.model.startswith("gpt-4")
        ):
            warnings.warn(
                "Temperature is ignored when reasoning is enabled. "
                "Set temperature to None to avoid this warning.",
                UserWarning,
                stacklevel=2,
            )
            temp = None

        # If rate limit error happens then just wait a minute and try again
        params = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "temperature": temp,
            "max_output_tokens": max_tokens,
            "store": False,
        }

        if reasoning is not None:
            params["reasoning"] = {"effort": reasoning}

        try:
            if output_structure is not None:
                try:
                    response = self.client.responses.parse(
                        **params,
                        text_format=output_structure,
                    )
                except ValidationError as e:
                    if "I'm sorry" in str(e):
                        raise AIRefusalError(
                            message="No reason given, however caused validation error of JSON"
                        ) from None
                    raise

            else:
                response = self.client.responses.create(
                    **params,
                )

        except openai.BadRequestError as e:
            raise AICallerFailedError(message=str(e)) from e

        if response.incomplete_details is not None:
            raise AICallerFailedError(
                message=f"Response was incomplete: {response.incomplete_details}"
            )

        # Track API costs
        if hasattr(response, "usage"):
            cached_tokens = 0
            if hasattr(response.usage, "input_tokens_details") and hasattr(
                response.usage.input_tokens_details, "cached_tokens"
            ):
                cached_tokens = response.usage.input_tokens_details.cached_tokens

            track_api_cost(
                self.model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                cached_tokens,
            )

        if raw_output:
            return response

        if output_structure is not None:
            return response.output_parsed

        return response.output_text


class AICaller:
    """High-level interface for querying AI models.

    Manages multiple model instances and routes queries to the appropriate model.
    """

    def __init__(self):
        """Initialize the AICaller with available models."""
        self.models = {
            name: OpenAICaller(openai_client, definition.api_model, definition.limit)
            for name, definition in MODEL_REGISTRY.items()
        }

    def query(
        self, system, user, temp=None, model="gpt-4", max_tokens=50_000, **kwargs
    ):
        """Query an AI model.

        Args:
            system: System prompt/instructions for the model.
            user: User message/query.
            temp: Temperature parameter (default: None).
            model: Model to use (default: "gpt-4").
            max_tokens: Maximum tokens in response (default: 16000).
            **kwargs: Additional arguments passed to the model's query method.

        Returns:
            The model's response.

        Raises:
            ModelNotFoundError: If the specified model is not available.
        """
        if model not in self.models:
            raise ModelNotFoundError(model, list(self.models.keys()))

        selected_model = self.models[model]

        return selected_model.query(system, user, temp, max_tokens=max_tokens, **kwargs)


ai_caller = AICaller()
