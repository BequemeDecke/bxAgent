"""
This test checks if the preparation node correctly prepares the workspace.
"""

import asyncio
import logging
import tempfile
from unittest import TestCase
from pathlib import Path
from langgraph.types import GraphOutput
from unittest.mock import Mock, patch

from bxagent.agents.workflow.nodes.preparation_node import (
    create_preparation_node,
)
from bxagent.agents.workflow.state import WorkflowState
from bxagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from bxagent.preparation import build_preparation_agent
from bxagent.validation import ValidationExecutor, implementations


class TestPreparationNode(TestCase):
    def setUp(self):
        self.preparation_agent = build_preparation_agent(
            validation_executor=ValidationExecutor(
                validations={
                    "workspace_operability": {
                        "validation": implementations.WorkspaceOperabilityValidation(),
                        "validation_schema": implementations.WorkspaceOperabilityValidationConfig,
                    },
                    "commands_installed": {
                        "validation": implementations.CommandInstalledValidation(),
                        "validation_schema": implementations.CommandInstalledValidationConfig,
                    },
                }
            )
        ).compile()

        self.call_preparation_node = create_preparation_node(self.preparation_agent)

    @patch("shutil.which")
    def test_preparation_node__invoke_subgraph(self, mock_which: Mock):
        mock_which.side_effect = lambda cmd: (
            f"/usr/bin/{cmd}" if cmd in ["git", "docker"] else None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            package_path = "de.example.bxagent"
            source_model_path = workspace_path / "source_model.txt"
            target_model_path = workspace_path / "target_model.txt"
            transformation_plan_path = workspace_path / "TRANSFORMATION.md"

            source_model_path.write_text("source model implementation")
            target_model_path.write_text("target model implementation")

            initial_state = WorkflowState(
                transformation_plan=None,
                workspace_path=workspace_path,
                transformation_package_path=package_path,
                source_model_path=source_model_path,
                target_model_path=target_model_path,
                required_commands=["git", "docker"],
            )

            output: WorkflowState = asyncio.run(
                self.call_preparation_node(initial_state)
            )
            logging.debug(f"Output state: {output}")

            self.assertTrue(
                transformation_plan_path.exists(),
                "The preparation node should create a TRANSFORMATION.md file in the workspace.",
            )
            self.assertIsNotNone(
                output.get("transformation_plan"),
                "The output state should contain a transformation plan.",
            )
            tp_data = output["transformation_plan"].data

            self.assertEqual(
                tp_data["source_model_implementation"],
                "source model implementation",
                "The transformation plan should contain the source model implementation.",
            )
            self.assertEqual(
                tp_data["target_model_implementation"],
                "target model implementation",
                "The transformation plan should contain the target model implementation.",
            )
