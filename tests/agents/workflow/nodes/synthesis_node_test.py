"""
This test checks if the synthesis node correctly utilizes the synthesis subagent to think about the transformation itself and think about how to implement the transformation.
It has to write the plan into the `TRANSFORMATION.md` file.
"""

from typing import TypedDict
from unittest import TestCase
from unittest.mock import Mock
from langchain.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START

from bxagent.agents.synthesis.output import SynthesisResponseFormat
from bxagent.agents.workflow.nodes.synthesis_node import (
    create_call_synthesis_agent_function,
)


class TestSynthesisNode(TestCase):
    def setUp(self):
        class DummyState(TypedDict):
            structured_response: SynthesisResponseFormat

        def generate_response(state: DummyState) -> DummyState:
            return {
                "structured_response": SynthesisResponseFormat(
                    implementation_instructions="Some instructions on how to implement the transformation."
                )
            }

        graph_builder = StateGraph(DummyState)
        graph_builder.add_node("synthesis", generate_response)
        graph_builder.add_edge(START, "synthesis")
        self.graph = graph_builder.compile()

    def test_synthesis_node__invoke_subgraph(self):
        call_sub = create_call_synthesis_agent_function(self.graph)

        result = call_sub(
            {
                "transformation_source_model_description": "A model that does X",
                "transformation_target_model_description": "A model that does Y",
                "iteration": 0,
                "latest_audit_runs": [],
            }
        )

        self.assertEqual(
            result["iteration"],
            1,
            "The iteration should be incremented by 1 after calling the synthesis agent.",
        )
        self.assertEqual(
            result["implementation_instructions"],
            "Some instructions on how to implement the transformation.",
            "The implementation instructions should match the output of the synthesis agent.",  
        )
        self.assertEqual(
            result["iteration"],
            1,
            "The iteration should be incremented by 1 after calling the synthesis agent.",
        )
