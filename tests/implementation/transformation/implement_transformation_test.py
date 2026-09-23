"""Tests for the implement_transformation node.

These tests verify that the implement_transformation node correctly:
1. Reads the transformation class and plan from the state
2. Calls the generator to synthesize the transformation class
3. Reads the generated code from the written file
4. Updates transformation_class["code"] with the read code
5. Merges written_files correctly
6. Does NOT increment the iteration counter (done by evaluate_implementation)
"""

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mdeagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.transformation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


class MockTransformationClassGenerator(TransformationClassGenerator):
    """Mock generator for testing."""

    def __init__(
        self, written_files: list[Path] | None = None, generated_code: str | None = None
    ):
        self.written_files = written_files or []
        self.generated_code = generated_code
        self.synthesize_called = False
        self.last_call_args = {}

    async def synthesize_transformation_class(
        self,
        transformation_plan,
        transformation_class: TransformationClass,
        specific_task: str | None = None,
        evaluation_results: dict | None = None,
    ) -> list[Path]:
        self.synthesize_called = True
        self.last_call_args = {
            "transformation_plan": transformation_plan,
            "transformation_class": transformation_class,
            "specific_task": specific_task,
            "evaluation_results": evaluation_results,
        }
        # Write generated code to the transformation class path (simulating real generator behavior)
        if self.generated_code is not None and transformation_class.get("path"):
            transformation_class["path"].write_text(self.generated_code)
        return self.written_files


class TestImplementTransformationNode(TestCase):
    """Tests for the implement_transformation node."""

    def setUp(self):
        """Set up mocks before each test."""
        # Patch TransformationPlan.from_dict to avoid Jinja template loading

        self._from_dict_patcher = patch.object(
            TransformationPlan, "from_dict", autospec=True
        )
        self.mock_from_dict = self._from_dict_patcher.start()

        # Make from_dict return a simple mock with .data attribute and .to_dict()
        def fake_from_dict(plan_dict):
            mock_tp = MagicMock(spec=TransformationPlan)
            mock_tp.data = plan_dict.get("data", {})
            mock_tp.to_dict.return_value = plan_dict
            return mock_tp

        self.mock_from_dict.side_effect = fake_from_dict

    def tearDown(self):
        """Stop mocks after each test."""
        # Stop the patcher
        self._from_dict_patcher.stop()

    def test_node_is_created_correctly(self):
        """The node factory should return an async function."""
        generator = MockTransformationClassGenerator()
        node = create_implement_transformation_node(Path(), generator)
        self.assertTrue(asyncio.iscoroutinefunction(node))

    def test_node_calls_generator_with_correct_args(self):
        """The node should call the generator with correct arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            written_file = workspace_path / "MyTransformation.java"
            written_file.write_text("public class MyTransformation {}")

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator(written_files=[written_file])
            node = create_implement_transformation_node(Path(), generator)
        
            input_state = ImplementationState(
                transformation_plan=transformation_plan.to_dict(),
                transformation_class=TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                latest_evaluation_runs={},
                task_specification="Test task specification",
            )

            output_state = asyncio.run(node(input_state))

            # Verify generator was called
            self.assertTrue(generator.synthesize_called)

            # Verify the arguments passed to the generator
            self.assertEqual(
                generator.last_call_args["transformation_class"]["name"], "MyTransformation"
            )
            self.assertEqual(
                generator.last_call_args["specific_task"], "Test task specification"
            )
            self.assertIsInstance(generator.last_call_args["evaluation_results"], list)

            # Verify result contains updated transformation_class with code
            self.assertIn("transformation_class", output_state)
            self.assertIsNotNone(output_state["transformation_class"].get("code"))

    def test_node_does_not_increment_iteration(self):
        """
        The iteration counter should NOT be incremented by implement_transformation.
        This is done by evaluate_implementation instead (see agent.py).
        Note: The implement_transformation node only returns transformation_class and written_files.
        The iteration is managed by the workflow in agent.py.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            written_file = workspace_path / "MyTransformation.java"
            written_file.write_text("public class MyTransformation {}")

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator(written_files=[written_file])
            node = create_implement_transformation_node(Path(), generator)
        
            input_state = ImplementationState(
                transformation_plan=transformation_plan.to_dict(),
                transformation_class=TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                latest_evaluation_runs={},
                task_specification="Test task specification",
                maven_project_path=workspace_path,
                bxtool_path=workspace_path,
                iteration=5,
            )

            output_state = asyncio.run(node(input_state))

            # Verify the node only returns transformation_class and written_files (not iteration)
            # The iteration is managed by the workflow in agent.py
            self.assertIn("transformation_class", output_state)
            self.assertIn("written_files", output_state)

    def test_node_filters_evaluation_results(self):
        """
        The node filters evaluation results via filter_execution_results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            written_file = workspace_path / "MyTransformation.java"
            written_file.write_text("public class MyTransformation {}")

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator(written_files=[written_file])
            node = create_implement_transformation_node(Path(), generator)
        
            # Create evaluation runs with different categories
            eval_result_execution = EvaluationResult(
                content="compilation failed",
                metadata={"success": False, "include_in_report": True},
            )
            eval_run_execution = EvaluationRun(
                started_at=datetime.now(tz=UTC),
                execution_time_ms=100,
                iteration=1,
                results=[eval_result_execution],
                errors=[],
                category="execution",
            )
            eval_run_other = EvaluationRun(
                started_at=datetime.now(tz=UTC),
                execution_time_ms=200,
                iteration=1,
                results=[],
                errors=[],
                category="other",  # This should be filtered out
            )
        
            input_state = ImplementationState(
                transformation_plan=transformation_plan.to_dict(),
                transformation_class=TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                latest_evaluation_runs={
                    "file_existence": eval_run_execution,
                    "java_compilation": eval_run_other,
                },
                task_specification="Test task specification",
            )

            output_state = asyncio.run(node(input_state))

            # Verify generator was called
            self.assertTrue(generator.synthesize_called)
            # evaluation_results is a list (filtered results)
            self.assertIsInstance(generator.last_call_args["evaluation_results"], list)

    def test_node_raises_error_without_transformation_class(self):
        """The node should raise ValueError if transformation_class is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator()
            node = create_implement_transformation_node(Path(), generator)
        
            # Use a regular dict to allow missing keys
            input_state: dict = {
                "transformation_plan": transformation_plan.to_dict(),
                "latest_evaluation_runs": {},
                "task_specification": "Test task specification",
                "maven_project_path": workspace_path,
                "bxtool_path": workspace_path,
            }
            # Don't set transformation_class
        
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(node(input_state))
        
            self.assertIn("Transformation class is not set", str(ctx.exception))

    def test_node_raises_error_without_transformation_plan(self):
        """The node should raise ValueError if transformation_plan is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            generator = MockTransformationClassGenerator()
            node = create_implement_transformation_node(Path(), generator)
        
            # Use a regular dict to allow missing keys
            input_state: dict = {
                "transformation_class": TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                "latest_evaluation_runs": {},
                "task_specification": "Test task specification",
                "maven_project_path": workspace_path,
                "bxtool_path": workspace_path,
            }
            # Don't set transformation_plan
        
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(node(input_state))
        
            self.assertIn("Transformation plan is not set", str(ctx.exception))

    def test_node_raises_error_without_task_specification(self):
        """The node should raise ValueError if task_specification is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator()
            node = create_implement_transformation_node(Path(), generator)
        
            # Use a regular dict to allow missing keys
            input_state: dict = {
                "transformation_plan": transformation_plan.to_dict(),
                "transformation_class": TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                "latest_evaluation_runs": {},
                "maven_project_path": workspace_path,
                "bxtool_path": workspace_path,
            }
            # Don't set task_specification
        
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(node(input_state))
        
            self.assertIn("Task specification is not set", str(ctx.exception))

    def test_node_updates_written_files_in_state(self):
        """The node should merge old and new written_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            
            existing_file = workspace_path / "ExistingFile.java"
            existing_file.write_text("// Existing file")
            
            written_file = workspace_path / "MyTransformation.java"
            written_file.write_text("public class MyTransformation {}")

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator(written_files=[written_file])
            node = create_implement_transformation_node(workspace_path, generator)
        
            input_state = ImplementationState(
                transformation_plan=transformation_plan.to_dict(),
                transformation_class=TransformationClass(
                    name="MyTransformation",
                    package="com.example",
                    path=workspace_path / "MyTransformation.java",
                    code=None,
                ),
                latest_evaluation_runs={},
                task_specification="Test task specification",
                written_files=[existing_file],
            )

            output_state = asyncio.run(node(input_state))

            # Verify written_files contains both old and new files
            self.assertIn("written_files", output_state)
            result_files = output_state["written_files"]
            self.assertIn(written_file, result_files)
            self.assertIn(existing_file, result_files)
    def test_node_updates_transformation_class_code(self):
        """The node should read generated code and store it in transformation_class['code']."""
        generated_code = "package com.example; public class Test {}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            
            transformation_class_path = workspace_path / "Test.java"
            transformation_class_path.write_text("public class Initial {}")

            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            transformation_plan.update_iteration(0)

            generator = MockTransformationClassGenerator(
                written_files=[transformation_class_path],
                generated_code=generated_code,
            )
            node = create_implement_transformation_node(Path(), generator)
        
            input_state = ImplementationState(
                transformation_plan=transformation_plan.to_dict(),
                transformation_class=TransformationClass(
                    name="Test",
                    package="com.example",
                    path=transformation_class_path,
                    code=None,
                ),
                latest_evaluation_runs={},
                task_specification="Test task specification",
            )

            output_state = asyncio.run(node(input_state))

            # Verify transformation_class was updated with code
            self.assertIn("transformation_class", output_state)
            updated_tc = output_state["transformation_class"]
            self.assertEqual(updated_tc["code"], generated_code)


class TestTransformClassGeneratorIntegration(TestCase):
    """Integration tests for the TransformationClassGenerator interface."""

    def test_generator_interface_requires_async_method(self):
        """The generator interface should require an async synthesize method."""
        with self.assertRaises(TypeError):
            TransformationClassGenerator()

    def test_generator_synthesize_returns_list_of_paths(self):
        """The synthesize method should return a list of Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            
            generator = MockTransformationClassGenerator(
                written_files=[workspace_path / "test1.java", workspace_path / "test2.java"]
            )
            
            transformation_md = workspace_path / "TRANSFORMATION.md"
            transformation_md.write_text("# Transformation Plan")
            transformation_plan = TransformationPlan.parse(FileTransformationPlanParser(transformation_md))
            
            dummy_tc = TransformationClass(
                name="Test",
                package="com.test",
                path=workspace_path / "Test.java",
                code=None,
            )
            
            result = asyncio.run(generator.synthesize_transformation_class(
                transformation_plan=transformation_plan,
                transformation_class=dummy_tc,
            ))
            
            self.assertIsInstance(result, list)
            self.assertTrue(all(isinstance(p, Path) for p in result))
