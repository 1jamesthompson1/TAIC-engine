import pytest
from pydantic import BaseModel, Field, ValidationError

from engine.utils.AICaller import ai_caller


@pytest.mark.parametrize(
    "model, user, expected_response",
    [
        pytest.param("gpt-4", "Hello this is a test", True, id="gpt-4"),
        pytest.param(
            "gpt-4",
            "Hello this is a test" * (10**6),
            False,
            id="gpt-4 over limit",
        ),
        pytest.param("gpt-5-mini", "Hello can you respond?", True, id="gpt-5-mini"),
    ],
)
def test_ai_caller(model, user, expected_response):
    response = ai_caller.query(model=model, system="", user=user, max_tokens=100)

    print(f"Response: '{response}'")
    assert isinstance(response, str) == expected_response
    return


def test_structured_output():
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
        print(response)
        country_summary = CountrySummary.model_validate_json(response)
        assert country_summary.country == "France"
        assert isinstance(country_summary.population, int)
        assert isinstance(country_summary.gdp, float)
    except ValidationError as e:
        pytest.fail(f"Response validation failed: {e}")


def test_varying_reasoning_levels():
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
        print(f"Response with reasoning={reasoning_level}:\n{response.output_text}\n")

        if reasoning_level == "none":
            assert response.usage.output_tokens_details.reasoning_tokens == 0

        else:
            reasoning_tokens.append(
                response.usage.output_tokens_details.reasoning_tokens
            )

    # Make sure that low reasoning uses fewer tokens than high reasoning
    if len(reasoning_tokens) == 3:
        assert (
            reasoning_tokens[0] < reasoning_tokens[1]
            and reasoning_tokens[1] < reasoning_tokens[2]
        ), "Low reasoning should use fewer tokens than high reasoning"


def test_ai_caller_invalid_model():
    with pytest.raises(Exception):
        ai_caller.query(
            model="gpt-6",
            system="",
            user="Hello this is a test",
            temp=0,
            max_tokens=100,
        )
