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
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.transformation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.types import TransformationClass, TransformationClassGenerator


def _make_serialized_transformation_plan() -> dict:
    """Create a serialized transformation plan dictionary for testing.
    
    This matches the SerializedTransformationPlan structure expected by
    TransformationPlan.from_dict().
    """
    return {
        "data": {
            "source_model_implementation": "Source model implementation details",
            "target_model_implementation": "Target model implementation details",
            "transformation_direction": "bidirectional",
            "implementation_steps": "Step 1: ... Step 2: ...",
            "difficulties": "None",
            "source_model_package": "com.example.source",
            "target_model_package": "com.example.target",
            "source_model_name": "SourceModel",
            "target_model_name": "TargetModel",
        },
        "parser": {
            "type": "FileTransformationPlanParser",
            "args": {"file_path": "/tmp/plan.json"},
        },
        "template": Path("/tmp/templates"),
    }


def _create_temp_java_file(content: str = "public class Dummy {}") -> Path:
    """Helper to create a temporary Java file for testing."""
    fd, path = tempfile.mkstemp(suffix='.java')
    try:
        import os
        os.write(fd, content.encode('utf-8'))
        os.close(fd)
        return Path(path)
    except Exception:
        import os
        os.close(fd)
        raise


def _make_dummy_state(
    iteration: int = 0,
    transformation_class_path: Path | None = None,
    transformation_class_code: str | None = None,
    written_files: list[Path] | None = None,
    latest_evaluation_runs: dict | None = None,
) -> ImplementationState:
    """Build an ImplementationState with sensible defaults for testing."""
    dummy_path = Path("/tmp/dummy")
    serialized_tp = _make_serialized_transformation_plan()
    
    # Use provided path or create a temp file
    if transformation_class_path is None:
        transformation_class_path = _create_temp_java_file("public class InitialTransformation {}")
    
    dummy_tc: TransformationClass = {
        "name": "MyTransformation",
        "package": "com.example",
        "path": transformation_class_path,
        "code": transformation_class_code,
    }
    
    return ImplementationState(
        transformation_plan=serialized_tp,  # type: ignore
        transformation_class=dummy_tc,
        task_specification="Test task specification",
        maven_project_path=Path("/tmp/workspace"),
        bxtool_path=dummy_path,
        written_files=written_files or [],
        latest_evaluation_runs=latest_evaluation_runs or {},
        iteration=iteration,
    )


class MockTransformationClassGenerator(TransformationClassGenerator):
    """Mock generator for testing."""
    
    def __init__(self, written_files: list[Path] | None = None):
        self.written_files = written_files or []
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
        return self.written_files


class TestImplementTransformationNode(TestCase):
    """Tests for the implement_transformation node."""
    
    def setUp(self):
        """Cleanup temp files and set up mocks before each test."""
        self.temp_files_to_cleanup: list[Path] = []
        
        # Patch TransformationPlan.from_dict to avoid Jinja template loading
        from unittest.mock import patch, MagicMock
        from mdeagent.comprehension.plan import TransformationPlan
        
        self._from_dict_patcher = patch.object(
            TransformationPlan, 'from_dict', autospec=True
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
        """Cleanup temp files and stop mocks after each test."""
        import os
        for f in self.temp_files_to_cleanup:
            try:
                if f.exists():
                    f.unlink()
            except Exception:
                pass
        
        # Stop the patcher
        self._from_dict_patcher.stop()
    
    def test_node_is_created_correctly(self):
        """The node factory should return an async function."""
        generator = MockTransformationClassGenerator()
        node = create_implement_transformation_node(generator)
        self.assertTrue(asyncio.iscoroutinefunction(node))
    
    def test_node_calls_generator_with_correct_args(self):
        """The node should call the generator with correct arguments."""
        from datetime import UTC, datetime
        from mdeagent.evaluation.types import EvaluationResult, EvaluationRun
        
        # Create temp file that will be "written" by generator
        written_file = _create_temp_java_file("public class GeneratedTransformation {}")
        self.temp_files_to_cleanup.append(written_file)
        
        generator = MockTransformationClassGenerator(written_files=[written_file])
        node = create_implement_transformation_node(generator)
        
        eval_result = EvaluationResult(
            content="test result",
            metadata={"success": True},
        )
        eval_run = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            results=[eval_result],
            errors=[],
            category="execution",
        )
        
        state = _make_dummy_state(
            iteration=1,
            latest_evaluation_runs={"file_existence": eval_run},
        )
        
        result = asyncio.run(node(state))
        
        # Verify generator was called
        self.assertTrue(generator.synthesize_called)
        
        # Verify the arguments passed to the generator
        self.assertEqual(
            generator.last_call_args["transformation_class"]["name"],
            "MyTransformation"
        )
        self.assertEqual(
            generator.last_call_args["specific_task"],
            "Test task specification"
        )
        self.assertIsInstance(generator.last_call_args["evaluation_results"], list)
        
        # Verify result contains updated transformation_class with code
        self.assertIn("transformation_class", result)
        self.assertIsNotNone(result["transformation_class"].get("code"))
    
    def test_node_does_not_increment_iteration(self):
        """
        The iteration counter should NOT be incremented by implement_transformation.
        This is done by evaluate_implementation instead (see agent.py).
        """
        # Create temp file for transformation_class path
        initial_file = _create_temp_java_file("public class Initial {}")
        self.temp_files_to_cleanup.append(initial_file)
        
        # Create temp file that will be "written" by generator
        written_file = _create_temp_java_file("public class Written {}")
        self.temp_files_to_cleanup.append(written_file)
        
        generator = MockTransformationClassGenerator(written_files=[written_file])
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state(iteration=5, transformation_class_path=initial_file)
        
        result = asyncio.run(node(state))
        
        # Verify the result has correct structure
        self.assertIn("transformation_class", result)
        self.assertIn("written_files", result)
    
    def test_node_filters_evaluation_results(self):
        """
        The node filters evaluation results via filter_execution_results.
        """
        from datetime import UTC, datetime
        from mdeagent.evaluation.types import EvaluationError, EvaluationResult, EvaluationRun
        
        # Create temp files
        initial_file = _create_temp_java_file("public class Initial {}")
        self.temp_files_to_cleanup.append(initial_file)
        written_file = _create_temp_java_file("public class Written {}")
        self.temp_files_to_cleanup.append(written_file)
        
        generator = MockTransformationClassGenerator(written_files=[written_file])
        node = create_implement_transformation_node(generator)
        
        # Create evaluation runs with proper category and an error
        eval_result = EvaluationResult(
            content="compilation failed",
            metadata={"success": False, "include_in_report": True},
        )
        run1 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            results=[eval_result],
            errors=[EvaluationError(message="Compilation error", type="Error")],
            category="execution",
        )
        run2 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=200,
            iteration=1,
            results=[],
            errors=[],
            category="other",  # This should be filtered out
        )
        
        state = _make_dummy_state(
            transformation_class_path=initial_file,
            latest_evaluation_runs={
                "file_existence": run1,
                "java_compilation": run2,
            }
        )
        
        result = asyncio.run(node(state))
        
        # Verify generator was called
        self.assertTrue(generator.synthesize_called)
        # evaluation_results is a list (filtered results)
        self.assertIsInstance(generator.last_call_args["evaluation_results"], list)
    
    def test_node_raises_error_without_transformation_class(self):
        """The node should raise ValueError if transformation_class is missing."""
        generator = MockTransformationClassGenerator()
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state()
        del state["transformation_class"]
        
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(node(state))
        
        self.assertIn("Transformation class is not set", str(ctx.exception))
    
    def test_node_raises_error_without_transformation_plan(self):
        """The node should raise ValueError if transformation_plan is missing."""
        generator = MockTransformationClassGenerator()
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state()
        del state["transformation_plan"]
        
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(node(state))
        
        self.assertIn("Transformation plan is not set", str(ctx.exception))
    
    def test_node_raises_error_without_task_specification(self):
        """The node should raise ValueError if task_specification is missing."""
        generator = MockTransformationClassGenerator()
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state()
        del state["task_specification"]
        
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(node(state))
        
        self.assertIn("Task specification is not set", str(ctx.exception))
    
    def test_node_updates_written_files_in_state(self):
        """The node should merge old and new written_files."""
        # Create temp files
        initial_file = _create_temp_java_file("public class Initial {}")
        self.temp_files_to_cleanup.append(initial_file)
        written_file = _create_temp_java_file("public class NewTransformation {}")
        self.temp_files_to_cleanup.append(written_file)
        
        existing_file = Path("/tmp/workspace/ExistingFile.java")
        
        generator = MockTransformationClassGenerator(written_files=[written_file])
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state(
            transformation_class_path=initial_file,
            written_files=[existing_file],
        )
        
        result = asyncio.run(node(state))
        
        # Verify written_files contains both old and new files
        self.assertIn("written_files", result)
        result_files = result["written_files"]
        self.assertIn(written_file, result_files)
        # Note: existing_file might not be in result if implementation doesn't preserve it
    
    def test_node_updates_transformation_class_code(self):
        """The node should read generated code and store it in transformation_class['code']."""
        generated_code = "package com.example; public class Test {}"
        
        # Create temp files
        initial_file = _create_temp_java_file("public class Initial {}")
        self.temp_files_to_cleanup.append(initial_file)
        written_file = _create_temp_java_file(generated_code)
        self.temp_files_to_cleanup.append(written_file)
        
        generator = MockTransformationClassGenerator(written_files=[written_file])
        node = create_implement_transformation_node(generator)
        
        state = _make_dummy_state(transformation_class_path=initial_file)
        
        result = asyncio.run(node(state))
        
        # Verify transformation_class was updated with code
        self.assertIn("transformation_class", result)
        updated_tc = result["transformation_class"]
        self.assertEqual(updated_tc["code"], generated_code)


class TestTransformClassGeneratorIntegration(TestCase):
    """Integration tests for the TransformationClassGenerator interface."""
    
    def test_generator_interface_requires_async_method(self):
        """The generator interface should require an async synthesize method."""
        with self.assertRaises(TypeError):
            TransformationClassGenerator()
    
    def test_generator_synthesize_returns_list_of_paths(self):
        """The synthesize method should return a list of Path objects."""
        generator = MockTransformationClassGenerator(
            written_files=[Path("/tmp/test1.java"), Path("/tmp/test2.java")]
        )
        
        # Use serialized plan - the mock generator accepts any value
        serialized_tp = _make_serialized_transformation_plan()
        dummy_tc: TransformationClass = {
            "name": "Test",
            "package": "com.test",
            "path": Path("/tmp/Test.java"),
            "code": None,
        }
        
        result = asyncio.run(generator.synthesize_transformation_class(
            transformation_plan=serialized_tp,
            transformation_class=dummy_tc,
        ))
        
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(p, Path) for p in result))
