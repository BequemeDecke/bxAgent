import asyncio
import tempfile
import logging

from unittest import TestCase
from unittest.mock import patch, Mock
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput
from pathlib import Path

from bxagent.preparation.agent import build_preparation_agent
from bxagent.preparation.state import PreparationState
from bxagent.validation import ValidationExecutor, implementations


class TestPreparationAgent(TestCase):
    def setUp(self):
        self.validation_executor = ValidationExecutor(
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

    def test_agent__construction(self):
        agent = build_preparation_agent(validation_executor=self.validation_executor)
        graph = agent.compile()

        self.assertIsInstance(
            graph,
            CompiledStateGraph,
            "The preparation agent should compile to a CompiledStateGraph.",
        )

    @patch("shutil.which")
    def test_agent__execution(self, mock_which: Mock):
        mock_which.side_effect = lambda cmd: (
            f"/usr/bin/{cmd}" if cmd in ["git", "docker"] else None
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            package_path = "de.example.bxagent"
            source_model_path = workspace_path / "source_model.txt"
            target_model_path = workspace_path / "target_model.txt"

            source_model_path.write_text("source model implementation")
            target_model_path.write_text("target model implementation")

            agent = build_preparation_agent(
                validation_executor=self.validation_executor
            )
            graph = agent.compile()
            initial_state = PreparationState(
                workspace_path=workspace_path,
                package_path=package_path,
                required_commands=["git", "docker"],
                source_model_path=source_model_path,
                target_model_path=target_model_path,
            )

            output: GraphOutput = asyncio.run(
                graph.ainvoke(input=initial_state, version="v2")
            )
            output_state: PreparationState = output.value
            logging.debug(f"Output state: {output_state}")
            
            validation_runs = output_state.get("latest_validation_runs", {})
            errors = []
            for validation_name, validation_run in validation_runs.items():
                if validation_run.errors:
                    errors.extend(validation_run.errors)

            self.assertEqual(
                errors,
                [],
                "All validations should pass with the mocked environment.",
            )
            self.assertTrue(
                output_state["workspace_path"].exists(),
                "The workspace path should exist.",
            )
