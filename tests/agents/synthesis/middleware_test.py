import unittest
import logging

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain.agents import create_agent
from langchain.tools import tool

from src.agents.synthesis import build_synthesis_agent

logger = logging.getLogger(__name__)

class TestSynthesisAgentMiddleware(unittest.TestCase):
    """Test the SynthesisAgentMiddleware by invoking it with a simple input and checking the output and state."""
    
    @staticmethod
    @tool
    def write_file(file_path: str, content: str):
        """Simulate the write_file tool by logging the file path and content."""
        logger.info(f"Simulating writing to file: {file_path} with content: {content}")

    def setUp(self):
        self.model = GenericFakeChatModel()
        self.agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt="You are a helpful assistant that writes files.",
        )

    def test_middleware_processing(self):
        