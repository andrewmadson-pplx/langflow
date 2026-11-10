from langchain_perplexity import ChatPerplexity
from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.io import DropdownInput, IntInput, SecretStrInput
from pydantic.v1 import SecretStr


class PerplexityComponent(LCModelComponent):
    display_name = "Perplexity"
    description = "Generate text using models on the Perplexity Agent API."
    documentation = "https://docs.perplexity.ai/docs/agent-api/models"
    icon = "Perplexity"
    name = "PerplexityModel"

    inputs = [
        *LCModelComponent.get_base_inputs(),
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=[
                "perplexity/sonar",
                "openai/gpt-5.5",
                "anthropic/claude-sonnet-4-6",
                "google/gemini-3.1-pro-preview",
            ],
            value="perplexity/sonar",
            combobox=True,
            info="Any model ID from the Perplexity Agent API model list.",
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Output Tokens",
            info="The maximum number of tokens to generate. Anthropic models require it.",
            value=4096,
        ),
        SecretStrInput(
            name="api_key",
            display_name="Perplexity API Key",
            info="The Perplexity API Key to use for the Perplexity model.",
            advanced=False,
            required=True,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        return ChatPerplexity(
            model=self.model_name,
            pplx_api_key=SecretStr(self.api_key).get_secret_value(),
            max_tokens=self.max_tokens or None,
            use_responses_api=True,
        )
