"""This test case tests the functionality of the workflow architecture approach.

This approach utilizes LangGraph to create a workflow where the planning agent checks the results and decides whether to continue or not.
The test case ensures that the workflow is executed correctly and that the planning agent can effectively manage the workflow based on the results obtained from the execution agent.
"""

import asyncio
import uuid
import logging
from pathlib import Path
from unittest import TestCase

from bxagent.agents.workflow.agent import build_workflow_agent
from bxagent.agents.workflow.state import WorkflowState
from bxagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan


class TestWorkflowApproach(TestCase):
    """Test case for the workflow architecture approach."""

    def setUp(self):
        # Build the workflow agent
        self.agent = build_workflow_agent().compile()

        # Create a unique workspace for the test
        self.workspace_path = Path(".bxagent-tests") / str(uuid.uuid4())
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created test workspace at {self.workspace_path}")


        # Set a default transformation package path
        self.transformation_package_path = "com.example.transformation"

    def test_workflow_execution(self):
        """Test the execution of the workflow."""
        input_state = WorkflowState(
            required_commands=["javac"],
            workspace_path=self.workspace_path,
            transformation_package_path=self.transformation_package_path,
        )

        result = asyncio.run(self.agent.ainvoke(input_state, version="v2"))

        # Check that the workflow execution was successful and that a transformation plan was generated
        self.check_preparation_phase()

    def check_preparation_phase(self):
        tp_path = self.workspace_path / "TRANSFORMATION.md"
        self.assertTrue(
            tp_path.exists(),
            "The transformation plan file should exist after the preparation phase.",
        )

        tp = TransformationPlan.parse(
            FileTransformationPlanParser(tp_path)
        )  # This should not raise an error
        tp_data = tp.data

        self.assertTrue(
            tp_data["source_model_package"] != "",
            "The source model package should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["target_model_package"] != "",
            "The target model package should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["transformation_direction"] != "",
            "The transformation direction should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["source_model_implementation"] != "",
            "The source model implementation should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["target_model_implementation"] != "",
            "The target model implementation should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["implementation_steps"] != "",
            "The implementation steps should be defined in the transformation plan.",
        )
        self.assertTrue(
            tp_data["difficulties"] != "",
            "The difficulties should be defined in the transformation plan.",
        )
