from langchain_perplexity import PerplexityEmbeddings
from lfx.base.embeddings.model import LCEmbeddingsModel
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, IntInput, SecretStrInput
from pydantic.v1 import SecretStr

TEXTS_PER_REQUEST = 100


class BatchedPerplexityEmbeddings(PerplexityEmbeddings):
    """PerplexityEmbeddings that embeds documents TEXTS_PER_REQUEST texts at a time.

    PerplexityEmbeddings sends every text in one request, and the Embeddings API
    rejects a request with more than 512 texts or 120,000 tokens.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            vectors.extend(super().embed_documents(texts[start : start + TEXTS_PER_REQUEST]))
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            vectors.extend(await super().aembed_documents(texts[start : start + TEXTS_PER_REQUEST]))
        return vectors


class PerplexityEmbeddingsComponent(LCEmbeddingsModel):
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

    def build_embeddings(self) -> Embeddings:
        if not self.api_key:
            msg = "Perplexity API Key is required"
            raise ValueError(msg)

        return BatchedPerplexityEmbeddings(
            api_key=SecretStr(self.api_key).get_secret_value(),
            model=self.model,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
