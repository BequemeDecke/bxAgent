from unittest import TestCase

from mdeagent.mapping import mde_to_tools
from mdeagent.state import MDEAgentState


class TestMDEToToolsMapping(TestCase):
    def test_mapping(self):
        state = MDEAgentState(required_tools=["tool1", "tool2"])
        evaluation_params = mde_to_tools(state)
        self.assertIn("tools", evaluation_params)
        self.assertEqual(evaluation_params["tools"], ["tool1", "tool2"])
