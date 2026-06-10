from unittest import TestCase
from unittest.mock import Mock, patch

from langchain.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.graph.state import CompiledStateGraph

from bxagent.config import Config, VariablesConfig, WorkspaceConfig
from bxagent.agents.coding.agent import build_coding_deep_agent


class TestCodingAgent(TestCase):
    @patch("bxagent.config.Config.get_instance")
    @patch("bxagent.models.build_coding_model")
    def test_coding_agent__builds_correctly_with_config(
        self, mock_build_coding_model, mock_get_instance
    ):
        mock_build_coding_model.return_value = GenericFakeChatModel(messages=iter([]))
        config = Mock(spec=Config)
        config.VARIABLES = VariablesConfig()
        config.WORKSPACE = WorkspaceConfig()

        mock_get_instance.return_value = config
        agent = build_coding_deep_agent()
        self.assertIsInstance(agent, CompiledStateGraph)
        self.assertEqual(mock_build_coding_model.call_count, 1)
        self.assertEqual(mock_get_instance.call_count, 1)
