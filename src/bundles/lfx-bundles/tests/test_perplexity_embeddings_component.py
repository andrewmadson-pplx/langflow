import base64
import json
import math

import httpx
import pytest
from langchain_perplexity import PerplexityEmbeddings
from lfx_bundles.perplexity import PerplexityEmbeddingsComponent
from lfx_bundles.perplexity import perplexity_embeddings as module
from pydantic import SecretStr

# Each text is a number n below 256 whose fake embedding is the int8 vector [1, n // 16, n % 16],
# so every text has its own direction and a swapped or misordered vector changes the result.
# Tests that use these 3-value vectors set the expected model width to 3 with the small_width fixture.
NEGATIVE = "negative"  # int8 values -3 and 4
ZERO = "zero"


def _fake_vector(text):
    if text == NEGATIVE:
        return [-3, 4, 0]
    if text == ZERO:
        return [0, 0, 0]
    number = int(text)
    return [1, number // 16, number % 16]


def _unit_vector(text):
    vector = _fake_vector(text)
    norm = math.sqrt(sum(value * value for value in vector))
    return pytest.approx([value / norm for value in vector] if norm else vector)


def _row(index, vector):
    return {"object": "embedding", "index": index, "embedding": _encoded(vector)}


class FakeEmbeddingsAPI:
    """Stand in for POST /v1/embeddings. Records each request and returns the rows in reverse index order."""

    def __init__(self, rows_for=None):
        self.requests = []
        self.rows_for = rows_for or (lambda texts: [_row(i, _fake_vector(t)) for i, t in enumerate(texts)][::-1])

    def __call__(self, request):
        self.requests.append(request)
        texts = json.loads(request.content)["input"]
        body = {"object": "list", "model": "m", "data": self.rows_for(texts)}
        return httpx.Response(200, json=body)

    def payloads(self):
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture
def small_width(monkeypatch):
    monkeypatch.setattr(module, "MODEL_DIMENSIONS", {"pplx-embed-v1-4b": 3, "pplx-embed-v1-0.6b": 3})


def _encoded(vector):
    return base64.b64encode(bytes(value % 256 for value in vector)).decode()


def _build(api, **inputs):
    inputs.setdefault("api_key", "test-key")  # pragma: allowlist secret
    embeddings = PerplexityEmbeddingsComponent(max_retries=0, **inputs).build_embeddings()
    transport = httpx.MockTransport(api)
    embeddings.client = embeddings.client.with_options(http_client=httpx.Client(transport=transport))
    embeddings.async_client = embeddings.async_client.with_options(http_client=httpx.AsyncClient(transport=transport))
    return embeddings


async def _embed(embeddings, texts, *, use_async):
    if use_async:
        return await embeddings.aembed_documents(texts)
    return embeddings.embed_documents(texts)


async def _embed_query(embeddings, text, *, use_async):
    if use_async:
        return await embeddings.aembed_query(text)
    return embeddings.embed_query(text)


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


@pytest.mark.usefixtures("small_width")
@pytest.mark.parametrize("use_async", [False, True])
async def test_request_sends_default_model_header_key_and_format(use_async):
    api = FakeEmbeddingsAPI()
    embeddings = _build(api, api_key=" test-key\n")  # pragma: allowlist secret

    await _embed(embeddings, ["1"], use_async=use_async)

    [request] = api.requests
    assert request.method == "POST"
    assert request.url.path == "/v1/embeddings"
    assert request.headers.get_list("X-Pplx-Integration") == ["langflow"]
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "pplx-embed-v1-4b"
    assert payload["encoding_format"] == "base64_int8"
    model_input = next(field for field in PerplexityEmbeddingsComponent.inputs if field.name == "model")
    assert model_input.options == ["pplx-embed-v1-4b", "pplx-embed-v1-0.6b"]


def test_build_embeddings_rejects_unsupported_model():
    component = PerplexityEmbeddingsComponent(
        api_key="test-key",  # pragma: allowlist secret
        model="pplx-embed-context-v1-4b",
    )

    with pytest.raises(ValueError, match="Unsupported Perplexity embedding model: pplx-embed-context-v1-4b"):
        component.build_embeddings()


def test_build_embeddings_unwraps_secret_str_key():
    secret = SecretStr(" test-key\n")  # pragma: allowlist secret
    embeddings = PerplexityEmbeddingsComponent(api_key=secret).build_embeddings()

    assert embeddings.pplx_api_key.get_secret_value() == "test-key"  # pragma: allowlist secret
    assert embeddings.client.api_key == "test-key"  # pragma: allowlist secret


@pytest.mark.parametrize("api_key", [None, "", "  ", SecretStr(""), SecretStr("  ")])
def test_build_embeddings_requires_api_key(api_key, monkeypatch):
    monkeypatch.setenv("PPLX_API_KEY", "server-key")  # pragma: allowlist secret
    component = PerplexityEmbeddingsComponent(api_key=api_key)

    with pytest.raises(ValueError, match="Perplexity API Key is required"):
        component.build_embeddings()


@pytest.mark.usefixtures("small_width")
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("count", "batch_sizes"),
    [(0, []), (99, [99]), (100, [100]), (101, [100, 1]), (200, [100, 100]), (250, [100, 100, 50])],
)
async def test_embed_documents_batches_and_orders_by_index(count, batch_sizes, use_async):
    api = FakeEmbeddingsAPI()
    embeddings = _build(api)
    texts = [str(number) for number in range(count)]

    vectors = await _embed(embeddings, texts, use_async=use_async)

    sent = [payload["input"] for payload in api.payloads()]
    assert [len(batch) for batch in sent] == batch_sizes
    assert [text for batch in sent for text in batch] == texts
    assert vectors == [_unit_vector(text) for text in texts]


@pytest.mark.usefixtures("small_width")
@pytest.mark.parametrize("use_async", [False, True])
async def test_embeddings_are_unit_length(use_async):
    embeddings = _build(FakeEmbeddingsAPI())

    negative, zero = await _embed(embeddings, [NEGATIVE, ZERO], use_async=use_async)
    assert negative == pytest.approx([-0.6, 0.8, 0.0])
    assert zero == [0.0, 0.0, 0.0]
    assert await _embed_query(embeddings, NEGATIVE, use_async=use_async) == pytest.approx([-0.6, 0.8, 0.0])


@pytest.mark.usefixtures("small_width")
@pytest.mark.parametrize(
    ("indices", "use_async"),
    [
        pytest.param([0, 0], False, id="duplicate"),
        pytest.param([0, 2], False, id="gap"),
        pytest.param([None, 0], False, id="null"),
        pytest.param([-1, 0], False, id="negative"),
        pytest.param([0], False, id="missing"),
        pytest.param([0, 1, 2], False, id="extra"),
        pytest.param([0, 0], True, id="duplicate-async"),
    ],
)
async def test_embed_documents_rejects_unexpected_indices(indices, use_async):
    rows = [_row(index, [1, 0, 0]) for index in indices]
    embeddings = _build(FakeEmbeddingsAPI(rows_for=lambda _texts: rows))

    with pytest.raises(ValueError, match="unexpected indices for 2 texts"):
        await _embed(embeddings, ["1", "2"], use_async=use_async)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(("model", "width"), [("pplx-embed-v1-4b", 2560), ("pplx-embed-v1-0.6b", 1024)])
async def test_embeddings_have_the_model_width(model, width, use_async):
    api = FakeEmbeddingsAPI(rows_for=lambda texts: [_row(i, [-3, 4] + [0] * (width - 2)) for i in range(len(texts))])
    embeddings = _build(api, model=model)

    [vector] = await _embed(embeddings, ["text"], use_async=use_async)

    assert len(vector) == width
    assert vector[:2] == pytest.approx([-0.6, 0.8])


@pytest.mark.parametrize(
    ("vector", "width", "use_async"),
    [
        # A base64_binary embedding for pplx-embed-v1-4b packs 2,560 bits into 320 bytes.
        pytest.param([128, 1] * 160, 320, False, id="packed-binary"),
        pytest.param([128, 1] * 160, 320, True, id="packed-binary-async"),
        pytest.param([1] * 2561, 2561, False, id="too-wide"),
    ],
)
async def test_embeddings_with_the_wrong_width_are_rejected(vector, width, use_async):
    embeddings = _build(FakeEmbeddingsAPI(rows_for=lambda _texts: [_row(0, vector)]))

    with pytest.raises(ValueError, match=f"{width} values for pplx-embed-v1-4b, expected 2560"):
        await _embed(embeddings, ["text"], use_async=use_async)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(None, "returned no data", id="null-data"),
        pytest.param([{"object": "embedding", "index": 0, "embedding": None}], "no embedding for input 0", id="null"),
        pytest.param([{"object": "embedding", "index": 0, "embedding": ""}], "no embedding for input 0", id="empty"),
    ],
)
def test_response_without_data_or_embedding_is_rejected(rows, message):
    embeddings = _build(FakeEmbeddingsAPI(rows_for=lambda _texts: rows))

    with pytest.raises(ValueError, match=message):
        embeddings.embed_documents(["text"])
