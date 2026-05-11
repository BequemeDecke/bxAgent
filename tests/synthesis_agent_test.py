import unittest
import logging

from langchain.messages import HumanMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

from src.agents.synthesis import build_synthesis_agent

logger = logging.getLogger(__name__)


class TestSynthesisAgent(unittest.TestCase):
    """Test the SynthesisAgent by invoking it with a simple input and checking the output and state."""

    def setUp(self):
        self.model = GenericFakeChatModel()
        self.agent = build_synthesis_agent(model=self.model)
        self.config = {"configurable": {"thread_id": "test_thread_id"}}
    