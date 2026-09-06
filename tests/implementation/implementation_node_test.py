import asyncio
import logging
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from mdeagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from mdeagent.implementation.node import (
    create_implementation_node,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)


class TestImplementationNode(TestCase):
    def setUp(self):
        self.fake_response: GraphOutput = GraphOutput(
            value=ImplementationState(
                transformation_md=None,
                task_specification="",
                bxtool_path=None,
                written_java_files=[Path("file1.java"), Path("file2.java")],
            )
        )

        async def fake_ainvoke(*args, **kwargs):
            return self.fake_response

        self.coding_agent = MagicMock(spec=CompiledStateGraph)
        self.coding_agent.ainvoke = MagicMock(side_effect=fake_ainvoke)

        self.call_implementation = create_implementation_node(self.coding_agent)

    def test_call_implementation_agent__invoke_coding_agent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            tp_file = workspace / "TRANSFORMATION.md"
            tp_file.touch()

            input_state = MDEAgentState(
                transformation_plan=TransformationPlan.parse(
                    FileTransformationPlanParser(tp_file)
                ),
                transformation_class_path=workspace / "TransformationClass.java",
                bxtool_path=workspace / "TransformationClassBxToolAdapter.java",
                written_files=[workspace / "existing_file.java"],
            )

            output_state: MDEAgentState = asyncio.run(
                self.call_implementation(input_state)
            )
            logger.debug(f"Output state: {output_state}")

            self.assertIn(
                "written_files",
                output_state,
                "Output state should contain 'written_files' key.",
            )
            self.assertIn(
                workspace / "existing_file.java",
                output_state["written_files"],
                "Existing file should be in the written files.",
            )
            self.assertIn(
                Path("file1.java"),
                output_state["written_files"],
                "file1.java should be in the written files.",
            )
            self.assertIn(
                Path("file2.java"),
                output_state["written_files"],
                "file2.java should be in the written files.",
            )

            self.coding_agent.ainvoke.assert_called_once()
