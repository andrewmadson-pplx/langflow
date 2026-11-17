import math

from langchain_perplexity import PerplexityEmbeddings
from lfx.base.embeddings.model import LCEmbeddingsModel
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, IntInput, SecretStrInput

INTEGRATION_HEADERS = {
    "X-Pplx-Integration": "langflow",
    "User-Agent": "langflow-perplexity-bundle",
}
TEXTS_PER_REQUEST = 100


def _unit_length(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class NormalizedPerplexityEmbeddings(PerplexityEmbeddings):
    """PerplexityEmbeddings that returns unit-length vectors and sends TEXTS_PER_REQUEST texts per request.

    The Embeddings API returns unnormalized vectors, which rank correctly only by cosine similarity.
    Unit-length vectors rank the same by cosine similarity, inner product and L2 distance, so they
    also work in vector stores that use L2 distance, such as the FAISS component.

    PerplexityEmbeddings sends every text in one request, and the Embeddings API rejects a request
    with more than 512 texts or 120,000 tokens.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            batch = super().embed_documents(texts[start : start + TEXTS_PER_REQUEST])
            vectors.extend(_unit_length(vector) for vector in batch)
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            batch = await super().aembed_documents(texts[start : start + TEXTS_PER_REQUEST])
            vectors.extend(_unit_length(vector) for vector in batch)
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
        api_key = (self.api_key or "").strip()
        if not api_key:
            msg = "Perplexity API Key is required"
            raise ValueError(msg)

        embeddings = NormalizedPerplexityEmbeddings(
            api_key=api_key,
            model=self.model,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )
        embeddings.client = embeddings.client.with_options(default_headers=INTEGRATION_HEADERS)
        embeddings.async_client = embeddings.async_client.with_options(default_headers=INTEGRATION_HEADERS)
        return embeddings
