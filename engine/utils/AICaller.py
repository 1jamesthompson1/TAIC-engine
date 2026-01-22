import os

import openai
import tiktoken
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

# Initialize clients only if API keys are available
openai_client = None

# For OpenAI, prioritize Azure OpenAI if credentials are available, otherwise use standard OpenAI
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_key = os.getenv("OPENAI_API_KEY")

if azure_api_key and azure_endpoint:
    # Use Azure OpenAI
    openai_client = AzureOpenAI(
        api_key=azure_api_key, api_version="2024-10-21", azure_endpoint=azure_endpoint
    )
    print("Using Azure!")
elif openai_api_key:
    # Use standard OpenAI
    openai_client = OpenAI(api_key=openai_api_key)


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

    def query(self, system, user, temp, n, max_tokens=1024):
        if self.client is None:
            raise ValueError(
                "OpenAI client is not available. Please set either OPENAI_API_KEY or both "
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables."
            )

        if self.check_query_above_limit(system + user):
            print("Too many tokens, not sending to OpenAI")
            return None

        # If rate limit error happens then just wait a minute and try again
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temp,
                n=n,
                max_tokens=max_tokens,
                seed=42,
            )
        except openai.BadRequestError as e:
            print(f"Too many tokens in request OpenAI declined:\n {e}")
            return None

        if n == 1:
            return completion.choices[0].message.content
        else:
            return [choice.message.content for choice in completion.choices]


class AICaller:
    def __init__(self):
        self.models = {
            "gpt-4": OpenAICaller(openai_client, "gpt-4o", 128_000),
        }

    def query(self, system, user, temp=1, model="gpt-4", n=1, max_tokens=1024):
        if model not in self.models:
            raise ValueError(
                f"Model {model} not found. Available models: {list(self.models.keys())}"
            )

        selected_model = self.models[model]

        return selected_model.query(system, user, temp, n, max_tokens)


ai_caller = AICaller()
