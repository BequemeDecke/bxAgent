import unittest

from langchain.tools import BaseTool
from bxagent.tools.validation.factory import create_validation_tools

class TestValidationToolFactory(unittest.TestCase):
    def test_create_validation_tools__defined(self):
        self.assertTrue(
            hasattr(create_validation_tools, "__call__"),
            "create_validation_tools should be a callable function.",
        )
        
    def test_create_validation_tools__returns_langchain_tool(self):
        tool = create_validation_tools({})
        self.assertTrue(
            isinstance(tool, list),
            "The created validation tools should be a list of BaseTool instances.",
        )
        self.assertTrue(
            all(isinstance(t, BaseTool) for t in tool),
            "All created validation tools should be instances of BaseTool.",
        )