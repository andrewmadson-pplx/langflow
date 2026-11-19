import base64
import math
import struct

from langchain_perplexity import PerplexityEmbeddings
from lfx.base.embeddings.model import LCEmbeddingsModel
from lfx.field_typing import Embeddings
from lfx.io import DropdownInput, IntInput, SecretStrInput

INTEGRATION_HEADERS = {"X-Pplx-Integration": "langflow"}
TEXTS_PER_REQUEST = 100
ENCODING_FORMAT = "base64_int8"
# Full output dimensions from https://docs.perplexity.ai/docs/embeddings. The component does not
# request fewer dimensions, so each base64_int8 embedding decodes to exactly this many values.
MODEL_DIMENSIONS = {"pplx-embed-v1-4b": 2560, "pplx-embed-v1-0.6b": 1024}


def _decode_int8(encoded: str) -> list[float]:
    raw = base64.b64decode(encoded)
    return [float(value) for value in struct.unpack(f"<{len(raw)}b", raw)]


def _unit_length(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def _vectors_in_input_order(response, texts: list[str], model: str) -> list[list[float]]:
    if response.data is None:
        msg = "Perplexity Embeddings API returned no data"
        raise ValueError(msg)
    rows = sorted(response.data, key=lambda row: -1 if row.index is None else row.index)
    if [row.index for row in rows] != list(range(len(texts))):
        msg = f"Perplexity Embeddings API returned embeddings with unexpected indices for {len(texts)} texts"
        raise ValueError(msg)
    expected = MODEL_DIMENSIONS.get(model)
    vectors = []
    for row in rows:
        if not row.embedding:
            msg = f"Perplexity Embeddings API returned no embedding for input {row.index}"
            raise ValueError(msg)
        vector = _decode_int8(row.embedding)
        if expected is not None and len(vector) != expected:
            msg = (
                f"Perplexity Embeddings API returned an embedding with {len(vector)} values for {model}, "
                f"expected {expected}"
            )
            raise ValueError(msg)
        vectors.append(_unit_length(vector))
    return vectors


class NormalizedPerplexityEmbeddings(PerplexityEmbeddings):
    """PerplexityEmbeddings that sends up to TEXTS_PER_REQUEST texts per request and normalizes nonzero vectors.

    PerplexityEmbeddings sends every text in one request and keeps the rows in response order. The
    Embeddings API takes at most 512 texts and 120,000 tokens per request, and 32K tokens per text.
    This class bounds the number of texts per request, not the tokens. It orders the rows by their
    index, as the Perplexity quickstart does. It requests base64_int8 explicitly and checks each
    decoded width against the model's dimensions, because a base64_binary embedding is also a
    base64 string and would decode to the wrong number of values.

    The API returns unnormalized vectors; use cosine similarity for the raw vectors. Nonzero
    unit-length vectors rank the same by cosine similarity, inner product and L2 distance, so they
    also work in vector stores that use L2 distance, such as the FAISS component.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            batch = texts[start : start + TEXTS_PER_REQUEST]
            response = self.client.embeddings.create(model=self.model, input=batch, encoding_format=ENCODING_FORMAT)
            vectors.extend(_vectors_in_input_order(response, batch, self.model))
        return vectors

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), TEXTS_PER_REQUEST):
            batch = texts[start : start + TEXTS_PER_REQUEST]
            response = await self.async_client.embeddings.create(
                model=self.model, input=batch, encoding_format=ENCODING_FORMAT
            )
            vectors.extend(_vectors_in_input_order(response, batch, self.model))
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
            options=list(MODEL_DIMENSIONS),
            value="pplx-embed-v1-4b",
        ),
        SecretStrInput(name="api_key", display_name="Perplexity API Key", required=True),
        IntInput(name="max_retries", display_name="Max Retries", advanced=True, value=6),
        IntInput(name="timeout", display_name="Request Timeout", advanced=True, value=120),
    ]

    def build_embeddings(self) -> Embeddings:
        api_key = self.api_key
        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()
        api_key = (api_key or "").strip()
        if not api_key:
            msg = "Perplexity API Key is required"
            raise ValueError(msg)
        if self.model not in MODEL_DIMENSIONS:
            msg = f"Unsupported Perplexity embedding model: {self.model}"
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
