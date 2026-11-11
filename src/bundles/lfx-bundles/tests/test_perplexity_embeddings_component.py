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
