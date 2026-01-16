import os
import warnings

import anthropic
import openai
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize clients only if API keys are available
openai_client = None
anthropic_client = None

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

if os.getenv("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


class BaseAICaller:
    def __init__(self, client, model, limit):
        self.client = client
        self.model = model
        self.limit = limit

    def get_tokens(self, texts):
        # No need to check for model as all models have the same encoding
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo-16k")
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
        max_tokens=16_000,
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

        if reasoning != "none" and temp is not None:
            warnings.warn(
                "Temperature is ignored when reasoning is enabled. "
                "Set temperature to None to avoid this warning.",
                UserWarning,
            )

        # If rate limit error happens then just wait a minute and try again
        params = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "temperature": temp if reasoning != "none" else None,
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

        if raw_output:
            return response

        return response.output_text


class AnthropicCaller(BaseAICaller):
    def __init__(self, client, model, limit):
        # Deprecation warning: Anthropic support will be removed in a future release
        warnings.warn(
            "AnthropicCaller is deprecated and will no longer be updated"
            "Please migrate to OpenAI or another supported provider.",
            DeprecationWarning,
        )
        super().__init__(client, model, limit)

    def query(self, system, user, temp, max_tokens=1024):
        if self.client is None:
            raise ValueError(
                "Anthropic client is not available. Please set the ANTHROPIC_API_KEY environment variable."
            )

        if self.check_query_above_limit(system + user):
            print("Too many tokens, not sending to Anthropic")
            return None

        completion = self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temp,
            max_tokens=max_tokens,
        )

        return completion.content[0].text


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
