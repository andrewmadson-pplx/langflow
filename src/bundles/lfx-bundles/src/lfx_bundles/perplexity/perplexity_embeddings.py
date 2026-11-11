from langchain_perplexity import PerplexityEmbeddings
from lfx.base.models.model import LCModelComponent
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, IntInput, Output, SecretStrInput
from pydantic.v1 import SecretStr


class PerplexityEmbeddingsComponent(LCModelComponent):
    display_name = "Perplexity Embeddings"
    description = "Generate embeddings using Perplexity embedding models."
    documentation = "https://docs.perplexity.ai/docs/embeddings/quickstart"
    icon = "Perplexity"
    name = "PerplexityEmbeddings"

    inputs = [
        DropdownInput(
            name="model",
            display_name="Model",
            advanced=False,
            options=["pplx-embed-v1-4b", "pplx-embed-v1-0.6b"],
            value="pplx-embed-v1-4b",
        ),
        SecretStrInput(name="api_key", display_name="Perplexity API Key", required=True),
        IntInput(name="max_retries", display_name="Max Retries", advanced=True, value=6),
        IntInput(name="timeout", display_name="Request Timeout", advanced=True, value=120),
    ]

    outputs = [
        Output(display_name="Embeddings", name="embeddings", method="build_embeddings"),
    ]

    def build_embeddings(self) -> Embeddings:
        if not self.api_key:
            msg = "Perplexity API Key is required"
            raise ValueError(msg)

        return PerplexityEmbeddings(
            api_key=SecretStr(self.api_key).get_secret_value(),
            model=self.model,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
