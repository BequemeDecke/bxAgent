import asyncio
import tempfile
from pathlib import Path
from typing import TypedDict
from unittest import TestCase

from langgraph.graph.state import END, START, StateGraph

from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    SerializedTransformationPlan,
    TransformationPlan,
)
from mdeagent.tracking import control_iteration
from mdeagent.util import with_transformation


class TestState(TypedDict):
    iteration: int
    transformation_plan: SerializedTransformationPlan
    codes: int


def create_dummy_node(return_code: int):
    async def dummy_node(state: TestState) -> TestState:
        new_codes = state.get("codes", 0) + return_code
        return TestState(codes=new_codes)

    return dummy_node


class TestIterationControl(TestCase):
    def setUp(self):
        graph = StateGraph(state_schema=TestState)
        graph.add_node(
            "one",
            with_transformation(create_dummy_node(return_code=1), control_iteration),
        )
        graph.add_node("two", create_dummy_node(return_code=2))

        graph.add_edge(START, "one")
        graph.add_edge("one", "two")
        graph.add_conditional_edges(
            "two", lambda state: str(state.get("codes", 0) >= 10), {
                "True": END,
                "False": "one"
            }
        )

        self.graph = graph.compile()

    def test_update_iteration_and_transformation_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(workspace / "TRANSFORMATION.md")
            )
            input_state = TestState(
                iteration=0, transformation_plan=transformation_plan.to_dict(), codes=0
            )
            output = asyncio.run(self.graph.ainvoke(input_state, version="v2"))

            self.assertEqual(output.value.get("iteration"), 4)
            self.assertEqual(output.value.get("transformation_plan").get("data").get("iteration"), 4)

    def test_update_iteration_without_transformation_plan(self):
        self.fail("Not implemented")
