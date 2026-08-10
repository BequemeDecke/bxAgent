"""This test case tests the functionality of the workflow architecture approach.

This approach utilizes LangGraph to create a workflow where the planning agent checks the results and decides whether to continue or not.
The test case ensures that the workflow is executed correctly and that the planning agent can effectively manage the workflow based on the results obtained from the execution agent.
"""

import asyncio
import datetime
import logging
from pathlib import Path
from unittest import TestCase

from mdagent.agents.workflow.agent import build_workflow_agent
from mdagent.agents.workflow.state import WorkflowState
from mdagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from mdagent.monitoring import build_langfuse_client

TEST_ENVIRONMENT = Path(".mdagent-tests")


class TestWorkflowApproach(TestCase):
    """Test case for the workflow architecture approach."""

    def setUp(self):
        # Create a unique workspace for the test
        self.workspace_path = (
            TEST_ENVIRONMENT
            / "test-executions"
            / datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        )
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created test workspace at {self.workspace_path}")

        # Check if the setup files exist
        self.setup_files = TEST_ENVIRONMENT / "setup-files"
        self.source_model_path = self.setup_files / "Families"
        self.target_model_path = self.setup_files / "Persons"
        if not self.source_model_path.exists() or not self.target_model_path.exists():
            self.fail(
                f"Setup files not found. Please ensure that {self.source_model_path} and {self.target_model_path} exist."
            )
        if len(list(self.source_model_path.glob("*.java"))) != 4:
            self.fail(
                f"Expected 4 source model files in {self.source_model_path}, but found {len(list(self.source_model_path.glob('*.java')))}."
            )
        if len(list(self.target_model_path.glob("*.java"))) != 3:
            self.fail(
                f"Expected 3 target model files in {self.target_model_path}, but found {len(list(self.target_model_path.glob('*.java')))}."
            )

        # Set a default transformation package path
        self.transformation_package_path = "com.example.transformation"

        # Set up Langfuse client for monitoring
        self.langfuse_client, self.langfuse_handler = build_langfuse_client()

        # Build the workflow agent
        self.agent = build_workflow_agent(self.workspace_path).compile()

    def test_workflow_execution(self):
        """Test the execution of the workflow."""
        input_state = WorkflowState(
            required_commands=["javac"],
            workspace_path=self.workspace_path,
            transformation_package_path=self.transformation_package_path,
            source_model_path=self.source_model_path,
            target_model_path=self.target_model_path,
        )

        result = asyncio.run(
            self.agent.ainvoke(
                input_state, version="v2", config={"callbacks": [self.langfuse_handler]}
            )
        )

        # Flush the Langfuse handler to ensure all logs are sent
        self.langfuse_client.flush()

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
