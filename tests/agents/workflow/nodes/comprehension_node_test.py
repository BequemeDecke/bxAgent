"""
This test checks if the comprehension node correctly utilizes the comprehension subagent to think about the transformation itself and think about how to implement the transformation.
It has to write the plan into the `TRANSFORMATION.md` file.
"""

from typing import TypedDict
from unittest import TestCase
from langgraph.graph import StateGraph, START

from bxagent.agents.workflow.nodes.comprehension_node import (
    create_call_comprehension_agent_function,
)


class TestComprehensionNode(TestCase):
    def setUp(self):
        class DummyState(TypedDict):
            pass

        def generate_response(state: DummyState) -> DummyState:
            return {}

        graph_builder = StateGraph(DummyState)
        graph_builder.add_node("comprehension", generate_response)
        graph_builder.add_edge(START, "comprehension")
        self.graph = graph_builder.compile()

    def test_comprehension_node__invoke_subgraph(self):
        call_sub = create_call_comprehension_agent_function(self.graph)

        result = call_sub(
            {
                "transformation_source_model_description": "A model that does X",
                "transformation_target_model_description": "A model that does Y",
                "iteration": 1,
                "latest_validation_runs": [],
            }
        )

        self.assertEqual(
            result["iteration"],
            2,
            "The iteration should be incremented by 1 after calling the comprehension agent.",
        )
