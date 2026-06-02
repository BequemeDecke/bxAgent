from unittest import TestCase
from langgraph.graph import StateGraph

from bxagent.agents.workflow.agent import build_workflow_agent


class TestWorkflowAgent(TestCase):
    def test_build_workflow_agent(self):
        agent = build_workflow_agent()
        self.assertIsInstance(agent, StateGraph)
