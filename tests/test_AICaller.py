"""Tests for the AICaller module."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from engine.AICaller import QueryTooLongError, ai_caller

REASONING_LEVELS_COUNT = 3


@pytest.mark.parametrize(
    "model, user, expected_response",
    [
        pytest.param("gpt-4", "Hello this is a test", True, id="gpt-4"),
        pytest.param("gpt-5-mini", "Hello can you respond?", True, id="gpt-5-mini"),
    ],
)
def test_ai_caller(model, user, expected_response):
    """Test basic AI caller functionality with different models."""
    response = ai_caller.query(model=model, system="", user=user, max_tokens=100)

    assert isinstance(response, str) == expected_response


def test_ai_over_token_limit():
    """Test that exceeding token limit raises an exception."""
    with pytest.raises(QueryTooLongError):
        ai_caller.query(
            model="gpt-4",
            system="",
            user="Hello this is a test" * (10**6),
            max_tokens=100,
        )


def test_structured_output():
    """Test structured output with pydantic models."""

    class CountrySummary(BaseModel):
        country: str = Field(..., description="Name of the country")
        population: int = Field(..., description="Population of the country")
        gdp: float = Field(..., description="GDP of the country in USD")

    response = ai_caller.query(
        model="gpt-5-mini",
        system="Your job is too provide structured data about countries in JSON format.",
        user="Provide a summary for the country France including its population and GDP.",
        max_tokens=4000,
        output_structure=CountrySummary,
    )

    assert response is not None, "Response should not be None"

    try:
        assert response.country == "France"
        assert isinstance(response.population, int)
        assert isinstance(response.gdp, float)
    except ValidationError as e:
        pytest.fail(f"Response validation failed: {e}")


def test_varying_reasoning_levels():
    """Test that different reasoning levels produce different token counts."""
    reasoning_tokens = []
    for reasoning_level in ["minimal", "low", "high"]:
        response = ai_caller.query(
            model="gpt-5-mini",
            system="",
            user="Explain the theory of relativity in simple terms.",
            reasoning=reasoning_level,
            max_tokens=1000,
            raw_output=True,
        )

        assert (
            response.output_text is not None
        ), f"Response should not be None for reasoning={reasoning_level}"

        if reasoning_level == "none":
            assert response.usage.output_tokens_details.reasoning_tokens == 0

        else:
            reasoning_tokens.append(
                response.usage.output_tokens_details.reasoning_tokens
            )

    if len(reasoning_tokens) == REASONING_LEVELS_COUNT:
        assert (
            reasoning_tokens[0] < reasoning_tokens[1]
            and reasoning_tokens[1] < reasoning_tokens[2]
        ), "Low reasoning should use fewer tokens than high reasoning"


def test_ai_caller_invalid_model():
    """Test that invalid model raises an exception."""
    with pytest.raises(Exception, match="model"):
        ai_caller.query(
            model="gpt-6",
            system="",
            user="Hello this is a test",
            temp=0,
            max_tokens=100,
        )
