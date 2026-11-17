import base64
from types import SimpleNamespace

import pytest
from langchain_perplexity import PerplexityEmbeddings
from lfx_bundles.perplexity import PerplexityEmbeddingsComponent


def test_build_embeddings_passes_inputs():
    component = PerplexityEmbeddingsComponent(
        api_key="test-key",  # pragma: allowlist secret
        model="pplx-embed-v1-0.6b",
        max_retries=2,
        timeout=30,
    )

    embeddings = component.build_embeddings()

    assert isinstance(embeddings, PerplexityEmbeddings)
    assert embeddings.model == "pplx-embed-v1-0.6b"
    assert embeddings.pplx_api_key.get_secret_value() == "test-key"  # pragma: allowlist secret
    assert embeddings.max_retries == 2
    assert embeddings.request_timeout == 30


@pytest.mark.parametrize("api_key", [None, ""])
def test_build_embeddings_requires_api_key(api_key, monkeypatch):
    monkeypatch.setenv("PPLX_API_KEY", "server-key")  # pragma: allowlist secret
    component = PerplexityEmbeddingsComponent(api_key=api_key)

    with pytest.raises(ValueError, match="Perplexity API Key is required"):
        component.build_embeddings()


def _fake_create(batch_sizes):
    """Stand in for embeddings.create: record each batch size and encode each text's number as two int8 values."""

    def create(**kwargs):
        texts = kwargs["input"]
        batch_sizes.append(len(texts))
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=base64.b64encode(bytes(divmod(int(text), 100))).decode()) for text in texts]
        )

    return create


def test_embed_documents_batches_requests_and_keeps_order():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret
    batch_sizes = []
    embeddings.client = SimpleNamespace(embeddings=SimpleNamespace(create=_fake_create(batch_sizes)))

    vectors = embeddings.embed_documents([str(number) for number in range(250)])

    assert batch_sizes == [100, 100, 50]
    assert vectors == [[float(number // 100), float(number % 100)] for number in range(250)]


async def test_aembed_documents_batches_requests_and_keeps_order():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret
    batch_sizes = []
    create = _fake_create(batch_sizes)

    async def acreate(**kwargs):
        return create(**kwargs)

    embeddings.async_client = SimpleNamespace(embeddings=SimpleNamespace(create=acreate))

    vectors = await embeddings.aembed_documents([str(number) for number in range(250)])

    assert batch_sizes == [100, 100, 50]
    assert vectors == [[float(number // 100), float(number % 100)] for number in range(250)]
