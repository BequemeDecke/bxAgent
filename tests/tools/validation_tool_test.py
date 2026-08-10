import unittest

from langchain.tools import BaseTool
from bxagent.tools.evaluation import create_evaluation_tools


class TestEvaluationToolFactory(unittest.TestCase):
    def test_create_evaluation_tools__defined(self):
        self.assertTrue(
            hasattr(create_evaluation_tools, "__call__"),
            "create_evaluation_tools should be a callable function.",
        )

    def test_create_evaluation_tools__returns_langchain_tool(self):
        tool = create_evaluation_tools({})
        self.assertTrue(
            isinstance(tool, list),
            "The created evaluation tools should be a list of BaseTool instances.",
        )
        self.assertTrue(
            all(isinstance(t, BaseTool) for t in tool),
            "All created evaluation tools should be instances of BaseTool.",
        )
