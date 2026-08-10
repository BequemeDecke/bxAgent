import asyncio
import logging
import tempfile
import shutil

from pathlib import Path
from unittest import TestCase

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from mdagent.preparation.agent import build_preparation_graph
from mdagent.preparation.state import ModelImplementation, PreparationState
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


class TestPreparationAgentIntegration(TestCase):
    def setUp(self):
        if shutil.which("mvn") is None:
            self.skipTest("Maven is not installed. Skipping integration tests.")

        self.evaluation_executor = EvaluationExecutor(
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

    def test_agent__construction(self):
        agent = build_preparation_graph(evaluation_executor=self.evaluation_executor)
        graph = agent.compile()

        self.assertIsInstance(
            graph,
            CompiledStateGraph,
            "The preparation agent should compile to a CompiledStateGraph.",
        )

    def test_agent__execution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir, "workspace")
            workspace_path.mkdir()
            models_path = Path(temp_dir, "models")
            models_path.mkdir()
            group_id = "de.example"
            artifact_id = "mdagent"

            (source_model_path, *_) = create_test_model_package(
                models_path, "Source"
            )
            (target_model_path, *_) = create_test_model_package(
                models_path, "Target"
            )

            agent = build_preparation_graph(
                evaluation_executor=self.evaluation_executor
            )
            graph = agent.compile()
            initial_state = PreparationState(
                workspace_path=workspace_path,
                group_id=group_id,
                artifact_id=artifact_id,
                required_commands=["mvn"],
                source_model=ModelImplementation(
                    name="Source",
                    path=source_model_path,
                    implementation=None,
                ),
                target_model=ModelImplementation(
                    name="Target",
                    path=target_model_path,
                    implementation=None,
                ),
                install_benchmarx=True,
            )

            output: GraphOutput = asyncio.run(
                graph.ainvoke(input=initial_state, version="v2")
            )
            output_state: PreparationState = output.value
            logging.debug(f"Output state: {output_state}")

            evaluation_runs = output_state.get("latest_evaluation_runs", {})
            errors = []
            for evaluation_name, evaluation_run in evaluation_runs.items():
                if evaluation_run.errors:
                    errors.extend(evaluation_run.errors)

            self.assertEqual(
                errors,
                [],
                "All evaluations should pass with the mocked environment.",
            )
            self.assertTrue(
                output_state["workspace_path"].exists(),
                "The workspace path should exist.",
            )

            transformation_plan = output_state.get("transformation_plan")
            self.assertIsNotNone(
                transformation_plan,
                "The transformation plan should be generated and included in the output state.",
            )
            self.assertEqual(
                transformation_plan.data.get("iteration"),
                0,
                "The transformation plan should have the correct iteration number.",
            )

            bxtool_path = output_state.get("bxtool_path")
            self.assertIsNotNone(
                bxtool_path,
                "The bxtool path should be set in the output state.",
            )
            self.assertTrue(
                bxtool_path.exists(),
                "The bxtool path should point to an existing file.",
            )

            benchmarx_path = output_state.get("benchmarx_path")
            self.assertIsNotNone(
                benchmarx_path,
                "The benchmarx path should be set in the output state.",
            )
            self.assertTrue(
                benchmarx_path.exists(),
                "The benchmarx path should point to an existing directory.",
            )
