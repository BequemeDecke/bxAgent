"""
This test checks if the comprehension node correctly utilizes the comprehension subagent to think about the transformation itself and think about how to implement the transformation.
It has to write the plan into the `TRANSFORMATION.md` file.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase

from langgraph.graph import END, START, StateGraph

from mdeagent.comprehension.node import (
    create_comprehension_node,
)
from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    TransformationPlan,
    TransformationPlanData,
)
from mdeagent.comprehension.state import ComprehensionState


class TestComprehensionNode(TestCase):
    def setUp(self):
        def generate_response(state: ComprehensionState) -> ComprehensionState:
            # Simulate the comprehension agent: increment iteration and update disk
            tp_obj = state["transformation_plan"]

            # Handle both TransformationPlan objects and raw dicts
            if hasattr(tp_obj, "to_dict"):
                tp_dict = tp_obj.to_dict()
            else:
                tp_dict = tp_obj

            if tp_dict and isinstance(tp_dict, dict) and "parser" in tp_dict:
                # Use from_dict to get the plan data without re-parsing from disk
                # (the file was written as rendered template, not parseable format)
                plan = TransformationPlan.from_dict(tp_dict)
                # Increment from the current plan iteration
                plan.data["iteration"] = plan.data.get("iteration", 0) + 1
                parser = FileTransformationPlanParser.from_dict(tp_dict["parser"])
                parser.save(str(plan))
                return_dict = plan.to_dict()
            else:
                return_dict = tp_dict if isinstance(tp_dict, dict) else tp_obj

            return {
                "transformation_plan": return_dict,
                "latest_evaluation_runs": {},
                "iteration": state["iteration"] + 1,
            }

        graph_builder = StateGraph(ComprehensionState)
        graph_builder.add_node("comprehension", generate_response)
        graph_builder.add_edge(START, "comprehension")
        graph_builder.add_edge("comprehension", END)
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

            result = asyncio.run(
                call_sub(
                    {
                        "transformation_plan": tp.to_dict(),
                        "iteration": 1,  # Pass the outer state's iteration so create_comprehension_node uses it
                    }
                )
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

        async def run_test():
            with self.assertRaises(ValueError) as context:
                await call_sub(
                    {
                        "latest_evaluation_runs": {},
                    }
                )

            self.assertIn(
                "The comprehension node requires a transformation plan in the state.",
                str(context.exception),
                "The comprehension node should raise a ValueError if the transformation plan is missing in the state.",
            )

        asyncio.run(run_test())
