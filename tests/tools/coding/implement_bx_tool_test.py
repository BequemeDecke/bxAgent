"""
Test for the implementation of the bx tool. The test should check if the tool correctly integrates the transformation logic into a bx tool using a provided template.

Two types of tests should be implemented:
1. Unit tests: These tests will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent, ensuring that the generated bx tool works correctly within the agent
"""

from unittest import TestCase
from unittest.mock import Mock
from langchain.chat_models import BaseChatModel

from bxagent.tools.coding.implement_bx_tool import create_implement_bx_tool_node, create_input_prompt
from bxagent.tools.coding.state import CodingAgentState

class TestImplementBxTool(TestCase):
    def setUp(self):
        mocked_llm = Mock(spec=BaseChatModel)
        mocked_llm_structured_output = Mock(spec=BaseChatModel)
        mocked_llm.with_structured_output.return_value = mocked_llm_structured_output
        # TODO: Define the output
        self.implement_bx_tool = create_implement_bx_tool_node(mocked_llm)
        
    def test_implement_bx_tool__return_bxtool_implementation(self):
        self.fail("Not implemented yet")
        
