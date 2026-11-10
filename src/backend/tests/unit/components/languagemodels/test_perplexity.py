import sys
from unittest.mock import MagicMock

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
        )
        assert model is mock_chat.return_value

    def test_build_model_unset_max_tokens(self, component_class, mocker):
        mock_chat = mocker.patch.object(_PERPLEXITY_MODULE, "ChatPerplexity", return_value=MagicMock())
        component = component_class(api_key="dummy-key", model_name="openai/gpt-5.5", max_tokens=0)

        component.build_model()

        _, kwargs = mock_chat.call_args
        assert kwargs["max_tokens"] is None
        assert kwargs["model"] == "openai/gpt-5.5"
