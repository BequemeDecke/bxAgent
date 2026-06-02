from unittest import TestCase

from langgraph.graph.state import StateGraph

from bxagent.agents.planning import build_planning_agent


class TestPlanningAgent(TestCase):
    def test_planning_agent__builds_correctly(self):
        agent = build_planning_agent()
        self.assertIsNotNone(agent, "Failed to build the planning agent.")
        self.assertIsInstance(agent, StateGraph, "The planning agent should be an instance of StateGraph.")
