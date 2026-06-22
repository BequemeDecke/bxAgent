"""
This test checks if the comprehension node correctly utilizes the comprehension subagent to think about the transformation itself and think about how to implement the transformation.
It has to write the plan into the `TRANSFORMATION.md` file.
"""

from pathlib import Path
from typing import TypedDict
from unittest import TestCase
from unittest.mock import MagicMock
from langgraph.graph import StateGraph, START

from bxagent.agents.workflow.nodes.comprehension_node import (
    create_comprehension_node,
)
from bxagent.comprehension.plan import (
    TransformationPlan,
    TransformationPlanData,
    TransformationPlanParser,
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
        self.parser = MagicMock(spec=TransformationPlanParser)
        self.transformation_plan = TransformationPlan(
            parser=self.parser, template_path=Path() / "templates"
        )
        self.transformation_plan.data = TransformationPlanData(
            source_model_package="com.example.source",
            target_model_package="com.example.target",
            iteration=1,
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties="",
            implementation_steps="",
        )

    def test_comprehension_node__invoke_subgraph(self):
        call_sub = create_comprehension_node(self.graph)

        result = call_sub(
            {
                "transformation_plan": self.transformation_plan,
            }
        )

        self.assertEqual(
            result.get("transformation_plan").data.get("iteration"),
            2,
            "The iteration should be incremented by 1 after calling the comprehension agent.",
        )

    def test_comprehension_node__missing_transformation_plan(self):
        call_sub = create_comprehension_node(self.graph)

        with self.assertRaises(ValueError) as context:
            call_sub(
                {
                    "latest_validation_runs": [],
                }
            )

        self.assertIn(
            "The comprehension node requires a transformation plan in the state.",
            str(context.exception),
            "The comprehension node should raise a ValueError if the transformation plan is missing in the state.",
        )
