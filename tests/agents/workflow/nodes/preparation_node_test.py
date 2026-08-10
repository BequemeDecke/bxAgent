"""
This test checks if the preparation node correctly prepares the workspace.
"""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

from mdagent.agents.workflow.nodes.preparation_node import (
    create_preparation_node,
)
from mdagent.agents.workflow.state import WorkflowState
from mdagent.preparation import build_preparation_graph
from mdagent.evaluation import EvaluationExecutor, implementations


def create_test_model_package(temp_dir: Path, package_name: str):
    package_path = temp_dir / package_name
    package_path.mkdir()

    source_file = package_path / f"{package_name}.java"
    source_register_file = package_path / f"{package_name}Register.java"
    source_package_file = package_path / f"{package_name}Package.java"
    source_factory_file = package_path / f"{package_name}Factory.java"

    source_file.write_text(f"public interface {package_name} {{ }}")
    source_register_file.write_text(f"public interface {package_name}Register {{ }}")
    source_package_file.write_text(f"public interface {package_name}Package {{ }}")
    source_factory_file.write_text(f"public interface {package_name}Factory {{ }}")
    return (
        package_path,
        source_file,
        source_register_file,
        source_package_file,
        source_factory_file,
    )


class TestPreparationNodeIntegration(TestCase):
    def setUp(self):
        if shutil.which("mvn") is None:
            self.skipTest("Maven is not installed. Skipping integration tests.")

        self.preparation_agent = build_preparation_graph(
            evaluation_executor=EvaluationExecutor(
                evaluations={
                    "workspace_operability": {
                        "evaluation": implementations.WorkspaceOperabilityEvaluation(),
                        "evaluation_schema": implementations.WorkspaceOperabilityEvaluationConfig,
                    },
                    "commands_installed": {
                        "evaluation": implementations.CommandInstalledEvaluation(),
                        "evaluation_schema": implementations.CommandInstalledEvaluationConfig,
                    },
                }
            )
        ).compile()

        self.call_preparation_node = create_preparation_node(self.preparation_agent)

    def test_preparation_node__invoke_subgraph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            workspace_path.mkdir()
            models_path = Path(temp_dir) / "models"
            models_path.mkdir()
            group_id = "de.example"
            artifact_id = "mdagent"
            package_path = f"{group_id}.{artifact_id}"
            transformation_plan_path = workspace_path / artifact_id / "TRANSFORMATION.md"

            (
                (source_model_path, source_file, *_),
                (target_model_path, target_file, *_),
            ) = (
                create_test_model_package(models_path, "Source"),
                create_test_model_package(models_path, "Target"),
            )

            initial_state = WorkflowState(
                transformation_plan=None,
                workspace_path=workspace_path,
                group_id=group_id,
                artifact_id=artifact_id,
                source_model_path=source_model_path,
                target_model_path=target_model_path,
                required_commands=["mvn", "git"],
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

            self.assertIn(
                source_file.read_text(),
                tp_data["source_model_implementation"],
                "The transformation plan should contain the source model implementation.",
            )
            self.assertIn(
                target_file.read_text(),
                tp_data["target_model_implementation"],
                "The transformation plan should contain the target model implementation.",
            )
            self.assertEqual(
                tp_data["source_model_package"],
                "Source",
                "The transformation plan should contain the correct source model package.",
            )
            self.assertEqual(
                tp_data["target_model_package"],
                "Target",
                "The transformation plan should contain the correct target model package.",
            )

            self.assertIsNotNone(
                output.get("bxtool_path"),
                "The output state should contain the path to the BxAgentJavaBxTool.java file.",
            )
            self.assertTrue(
                output.get("bxtool_path").exists(),
                "The preparation node should create a BxAgentJavaBxTool.java file in the package path.",
            )
