"""
This test checks if the comprehension node correctly utilizes the comprehension subagent to think about the transformation itself and think about how to implement the transformation.
It has to write the plan into the `TRANSFORMATION.md` file.
"""

import tempfile
from pathlib import Path
from typing import TypedDict
from unittest import TestCase

from langgraph.graph import START, StateGraph

from mdeagent.comprehension.node import (
    create_comprehension_node,
)
from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    TransformationPlan,
    TransformationPlanData,
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
        self.transformation_plan_data = TransformationPlanData(
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

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser = FileTransformationPlanParser(temp_path / "TRANSFORMATION.md")
            tp = TransformationPlan.parse(parser)
            tp.data = self.transformation_plan_data
            parser.save(str(tp))

            result = call_sub(
                {
                    "transformation_plan": tp.to_dict(),
                }
            )

            serialized_tp = result.get("transformation_plan")
            self.assertIsInstance(
                serialized_tp,
                dict,
                "The comprehension node should return a dictionary representing the transformation plan.",
            )
            tp = TransformationPlan.from_dict(serialized_tp)
            self.assertEqual(
                tp.data.get("iteration"),
                2,
                "The iteration should be incremented by 1 after calling the comprehension agent.",
            )

    def test_comprehension_node__missing_transformation_plan(self):
        call_sub = create_comprehension_node(self.graph)

        with self.assertRaises(ValueError) as context:
            call_sub(
                {
                    "latest_evaluation_runs": [],
                }
            )

        self.assertIn(
            "The comprehension node requires a transformation plan in the state.",
            str(context.exception),
            "The comprehension node should raise a ValueError if the transformation plan is missing in the state.",
        )
