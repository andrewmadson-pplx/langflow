import base64
import math
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


def test_build_embeddings_strips_whitespace_around_api_key():
    component = PerplexityEmbeddingsComponent(api_key=" test-key\n")  # pragma: allowlist secret

    embeddings = component.build_embeddings()

    assert embeddings.pplx_api_key.get_secret_value() == "test-key"  # pragma: allowlist secret
    assert embeddings.client.api_key == "test-key"  # pragma: allowlist secret


def test_build_embeddings_sets_integration_headers_on_both_clients():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret

    for client in (embeddings.client, embeddings.async_client):
        assert client.default_headers["X-Pplx-Integration"] == "langflow"
        assert client.default_headers["User-Agent"] == "langflow-perplexity-bundle"


@pytest.mark.parametrize("api_key", [None, "", "  "])
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


def _unit_vector(number):
    high, low = divmod(number, 100)
    length = math.hypot(high, low)
    return pytest.approx([high / length, low / length])


def test_embed_documents_batches_requests_and_keeps_order():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret
    batch_sizes = []
    embeddings.client = SimpleNamespace(embeddings=SimpleNamespace(create=_fake_create(batch_sizes)))

    vectors = embeddings.embed_documents([str(number) for number in range(1, 251)])

    assert batch_sizes == [100, 100, 50]
    assert vectors == [_unit_vector(number) for number in range(1, 251)]


async def test_aembed_documents_batches_requests_and_keeps_order():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret
    batch_sizes = []
    create = _fake_create(batch_sizes)

    async def acreate(**kwargs):
        return create(**kwargs)

    embeddings.async_client = SimpleNamespace(embeddings=SimpleNamespace(create=acreate))

    vectors = await embeddings.aembed_documents([str(number) for number in range(1, 251)])

    assert batch_sizes == [100, 100, 50]
    assert vectors == [_unit_vector(number) for number in range(1, 251)]


async def test_embeddings_are_unit_length():
    embeddings = PerplexityEmbeddingsComponent(api_key="test-key").build_embeddings()  # pragma: allowlist secret
    create = _fake_create([])

    async def acreate(**kwargs):
        return create(**kwargs)

    embeddings.client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    embeddings.async_client = SimpleNamespace(embeddings=SimpleNamespace(create=acreate))

    assert embeddings.embed_documents(["304", "0"]) == [pytest.approx([0.6, 0.8]), [0.0, 0.0]]
    assert embeddings.embed_query("304") == pytest.approx([0.6, 0.8])
    assert await embeddings.aembed_documents(["304", "0"]) == [pytest.approx([0.6, 0.8]), [0.0, 0.0]]
    assert await embeddings.aembed_query("304") == pytest.approx([0.6, 0.8])
