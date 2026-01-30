import os
import warnings
from threading import Lock

import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table

load_dotenv()

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

# Pricing per 1M tokens in USD
_PRICING = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-4o": {"input": 5.00, "cached_input": 0.50, "output": 15.00},
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
    """Track API costs for monitoring purposes."""
    global _api_costs

    if model not in _PRICING:
        raise ValueError(f"Pricing for model {model} is not defined.")

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
    """Get current API cost tracking data."""
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
    """Reset cost tracking (useful for test isolation)."""
    global _api_costs
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
    def __init__(self, client, model, limit):
        self.client = client
        self.model = model
        self.limit = limit

    def get_tokens(self, texts):
        # No need to check for model as all models have the same or broadly similar tokenization
        enc = tiktoken.encoding_for_model("gpt-5")
        return [len(enc.encode(text)) for text in texts]

    def check_query_above_limit(self, query):
        return sum(self.get_tokens([query])) > self.limit


class OpenAICaller(BaseAICaller):
    def __init__(self, client, model, limit):
        super().__init__(client, model, limit)

    def query(
        self,
        system,
        user,
        temp,
        max_tokens=32_000,
        output_structure=None,
        reasoning=None,
        raw_output=False,
    ):
        if self.client is None:
            raise ValueError(
                "OpenAI client is not available. Please set either OPENAI_API_KEY or both "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
            )

        if self.check_query_above_limit(system + user):
            print("Too many tokens, not sending to OpenAI")
            return None

        if (
            reasoning != "none"
            and temp is not None
            and not self.model.startswith("gpt-4")
        ):
            warnings.warn(
                "Temperature is ignored when reasoning is enabled. "
                "Set temperature to None to avoid this warning.",
                UserWarning,
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
                response = self.client.responses.parse(
                    **params,
                    text_format=output_structure,
                )
            else:
                response = self.client.responses.create(
                    **params,
                )

        except openai.BadRequestError as e:
            print(f"OpenAI declined:\n {e}")
            return None

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
    def __init__(self):
        self.models = {
            "gpt-4": OpenAICaller(openai_client, "gpt-4o", 128_000),
            "gpt-5-mini": OpenAICaller(openai_client, "gpt-5-mini", 400_000),
        }

    def query(
        self, system, user, temp=None, model="gpt-4", max_tokens=16_000, **kwargs
    ):
        if model not in self.models:
            raise ValueError(
                f"Model {model} not found. Available models: {list(self.models.keys())}"
            )

        selected_model = self.models[model]

        return selected_model.query(system, user, temp, max_tokens=max_tokens, **kwargs)


ai_caller = AICaller()
