import json
import sys
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("lfx_bundles")

from lfx.components.perplexity.perplexity import PerplexityComponent
from lfx.custom.custom_component.component import Component
from lfx.custom.utils import build_custom_component_template
from lfx.inputs.inputs import DropdownInput, IntInput, SecretStrInput

from tests.base import ComponentTestBaseWithoutClient

# The component moved into lfx-bundles, so ``perplexity.py`` is reachable under several
# module identities (the ``lfx.components.perplexity.perplexity`` shim, the canonical
# ``lfx_bundles.perplexity.perplexity``, and the runtime ``_lfx_ext.*`` ext-loader copy).
# Patch on the module the imported class actually lives in: a fixed string target like
# "lfx.components.perplexity.perplexity.ChatPerplexity" misses (mock "called 0 times")
# when an earlier test in the suite causes the class to resolve to a different identity.
_PERPLEXITY_MODULE = sys.modules[PerplexityComponent.__module__]


class TestPerplexityComponent(ComponentTestBaseWithoutClient):
    @pytest.fixture
    def component_class(self):
        return PerplexityComponent

    @pytest.fixture
    def default_kwargs(self):
        return {
            "api_key": "dummy-key",  # pragma: allowlist secret
            "model_name": "perplexity/sonar",
            "max_tokens": 50,
        }

    @pytest.fixture
    def file_names_mapping(self):
        # No historical version snapshots to validate this component against.
        return []

    @pytest.fixture
    def skipped_outputs(self):
        return {
            "text_output": "sends the prompt to a live LLM",
        }

    def test_initialization(self, component_class):
        component = component_class()
        assert component.display_name == "Perplexity"
        assert component.description == "Generate text using models on the Perplexity Agent API."
        assert component.icon == "Perplexity"
        # The class name / identifier must remain stable: it keys saved flows.
        assert component.name == "PerplexityModel"

    def test_inputs(self):
        inputs = {inp.name: inp for inp in PerplexityComponent().inputs}
        assert isinstance(inputs["model_name"], DropdownInput)
        assert isinstance(inputs["api_key"], SecretStrInput)
        assert isinstance(inputs["max_tokens"], IntInput)
        assert inputs["api_key"].required is True
        assert inputs["model_name"].value == "perplexity/sonar"
        assert inputs["model_name"].combobox is True
        assert inputs["model_name"].options == [
            "perplexity/sonar",
            "openai/gpt-5.5",
            "anthropic/claude-sonnet-5-5",
            "google/gemini-4-argon",
        ]
        # No default: max_output_tokens is optional for every Agent API model.
        assert inputs["max_tokens"].value == 0
        # temperature, top_p and n stay declared, hidden, so that a flow saved with them
        # updates without a breaking change.
        for name in ("temperature", "top_p", "n"):
            assert inputs[name].show is False

    def test_template(self, default_kwargs):
        component = PerplexityComponent(**default_kwargs)
        frontend_node, _ = build_custom_component_template(Component(_code=component._code))
        assert isinstance(frontend_node, dict)
        assert "template" in frontend_node
        input_names = [inp["name"] for inp in frontend_node["template"].values() if isinstance(inp, dict)]
        for expected in ["model_name", "api_key", "max_tokens"]:
            assert expected in input_names

    def test_build_model(self, component_class, default_kwargs, mocker):
        # Regression guard for the langchain-community 0.4.2 migration: ChatPerplexity
        # moved out of langchain_community.chat_models into the standalone
        # langchain_perplexity package. build_model must construct that class with the
        # inputs mapped onto its kwargs.
        mock_chat = mocker.patch.object(_PERPLEXITY_MODULE, "ChatPerplexity", return_value=MagicMock())
        component = component_class(**default_kwargs)

        model = component.build_model()

        mock_chat.assert_called_once_with(
            model="perplexity/sonar",
            pplx_api_key="dummy-key",
            max_tokens=50,
            use_responses_api=True,
            model_kwargs={
                "extra_headers": {
                    "X-Pplx-Integration": "langflow",
                    "User-Agent": "langflow-perplexity-bundle",
                },
                "tools": [{"type": "web_search"}],
            },
        )
        assert model is mock_chat.return_value

    def test_build_model_unset_max_tokens(self, component_class, mocker):
        mock_chat = mocker.patch.object(_PERPLEXITY_MODULE, "ChatPerplexity", return_value=MagicMock())
        component = component_class(api_key="dummy-key", model_name="openai/gpt-5.5", max_tokens=0)

        component.build_model()

        _, kwargs = mock_chat.call_args
        assert kwargs["max_tokens"] is None
        assert kwargs["model"] == "openai/gpt-5.5"

    @pytest.mark.parametrize(
        "retired_model",
        [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
        ],
    )
    def test_build_model_replaces_retired_model(self, component_class, mocker, retired_model):
        # A flow saved before the move to the Agent API holds one of these IDs, plus the
        # three inputs the component no longer sends.
        mock_chat = mocker.patch.object(_PERPLEXITY_MODULE, "ChatPerplexity", return_value=MagicMock())
        component = component_class(
            api_key="dummy-key",  # pragma: allowlist secret
            model_name=retired_model,
            max_tokens=0,
            temperature=0.75,
            top_p=0.9,
            n=2,
        )

        component.build_model()

        _, kwargs = mock_chat.call_args
        assert kwargs["model"] == "perplexity/sonar"
        assert not {"temperature", "top_p", "n"} & set(kwargs)

    def test_request_reaches_agent_api(self, component_class, default_kwargs):
        # Build the real ChatPerplexity and capture the HTTP request it sends.
        from perplexity import Perplexity

        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 0,
                    "model": "perplexity/sonar",
                    "status": "completed",
                    "output": [
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                },
            )

        model = component_class(**default_kwargs).build_model()
        model.client = Perplexity(
            api_key="dummy-key",  # pragma: allowlist secret
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = model.invoke("hello")

        assert result.text == "ok"
        assert len(requests) == 1
        request = requests[0]
        assert request.url.path == "/v1/responses"
        assert request.headers["X-Pplx-Integration"] == "langflow"
        assert request.headers["User-Agent"] == "langflow-perplexity-bundle"
        body = json.loads(request.content)
        assert body["model"] == "perplexity/sonar"
        assert body["max_output_tokens"] == 50
        assert body["tools"] == [{"type": "web_search"}]
        assert not {"temperature", "top_p", "n", "max_tokens"} & set(body)
