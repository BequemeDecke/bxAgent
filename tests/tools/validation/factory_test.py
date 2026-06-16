import unittest

from langchain.tools import BaseTool
from bxagent.tools.validation.factory import create_audit_tools

class TestAuditToolFactory(unittest.TestCase):
    def test_create_audit_tools__defined(self):
        self.assertTrue(
            hasattr(create_audit_tools, "__call__"),
            "create_audit_tools should be a callable function.",
        )
        
    def test_create_audit_tools__returns_langchain_tool(self):
        tool = create_audit_tools({})
        self.assertTrue(
            isinstance(tool, list),
            "The created audit tools should be a list of BaseTool instances.",
        )
        self.assertTrue(
            all(isinstance(t, BaseTool) for t in tool),
            "All created audit tools should be instances of BaseTool.",
        )