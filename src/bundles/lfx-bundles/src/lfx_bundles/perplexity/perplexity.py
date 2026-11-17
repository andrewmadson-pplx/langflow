from langchain_perplexity import ChatPerplexity
from lfx.base.models.model import LCModelComponent
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.io import DropdownInput, FloatInput, IntInput, SecretStrInput, SliderInput
from pydantic.v1 import SecretStr

DEFAULT_MODEL = "perplexity/sonar"

# The model IDs this component offered before it moved to the Agent API. The API rejects
# all seven, so a flow saved with one of them runs on DEFAULT_MODEL.
RETIRED_MODELS = frozenset(
    {
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-huge-128k-online",
        "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-sonar-large-128k-chat",
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct",
    }
)


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
                DEFAULT_MODEL,
                "openai/gpt-5.5",
                "anthropic/claude-sonnet-5-5",
                "google/gemini-4-argon",
            ],
            value=DEFAULT_MODEL,
            combobox=True,
            info="Any model ID from the Perplexity Agent API model list.",
        ),
        IntInput(name="max_tokens", display_name="Max Output Tokens", info="The maximum number of tokens to generate."),
        SecretStrInput(
            name="api_key",
            display_name="Perplexity API Key",
            info="The Perplexity API Key to use for the Perplexity model.",
            advanced=False,
            required=True,
        ),
        # temperature, top_p and n are kept, hidden, so that flows saved with them update
        # without a breaking change. None of them is sent: langchain-perplexity drops
        # temperature and top_p on the Agent API route, and the Agent API rejects n.
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.75,
            range_spec=RangeSpec(min=0, max=2, step=0.05),
            info="Deprecated. Not sent to the Agent API.",
            show=False,
        ),
        FloatInput(
            name="top_p",
            display_name="Top P",
            info="Deprecated. Not sent to the Agent API.",
            advanced=True,
            show=False,
        ),
        IntInput(
            name="n",
            display_name="N",
            info="Deprecated. Not sent to the Agent API.",
            advanced=True,
            show=False,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        return ChatPerplexity(
            model=DEFAULT_MODEL if self.model_name in RETIRED_MODELS else self.model_name,
            pplx_api_key=SecretStr(self.api_key).get_secret_value(),
            max_tokens=self.max_tokens or None,
            use_responses_api=True,
        )
