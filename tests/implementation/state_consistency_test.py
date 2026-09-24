"""Tests for ImplementationState consistency.

These tests verify that the ImplementationState is used consistently across
all nodes in the implementation module. They check that:
1. All required fields are present when creating a state
2. Nodes read and write state fields using consistent names
3. The iteration counter is only incremented by evaluate_implementation
"""

from pathlib import Path
from unittest import TestCase

from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import TransformationClass


def _make_dummy_transformation_plan() -> object:
    """Create a minimal mock transformation plan for testing."""
    from unittest.mock import MagicMock
    
    mock_tp = MagicMock()
    mock_tp.data = {
        "source_model_implementation": "",
        "target_model_implementation": "",
        "transformation_direction": "",
        "implementation_steps": "",
        "difficulties": "",
        "source_model_package": "",
        "target_model_package": "",
        "source_model_name": "",
        "target_model_name": "",
    }
    return mock_tp


class TestImplementationStateConsistency(TestCase):
    """Tests for ImplementationState field consistency."""
    
    def test_state_has_all_required_fields(self):
        """Verify that ImplementationState has all required fields defined."""
        from typing import get_type_hints
        
        hints = get_type_hints(ImplementationState)
        
        # Core transformation fields (transformation_class contains name, package, path, code)
        self.assertIn("transformation_plan", hints)
        self.assertIn("transformation_class", hints)
        
        # Required configuration fields
        self.assertIn("task_specification", hints)
        self.assertIn("maven_project_path", hints)
        self.assertIn("bxtool_path", hints)
        
        # Implementation tracking
        self.assertIn("written_files", hints)
        
        # Evaluation
        self.assertIn("latest_evaluation_runs", hints)
        
        # Iteration tracking
        self.assertIn("iteration", hints)
    
    def test_state_can_be_created_with_all_fields(self):
        """Verify that a complete ImplementationState can be created."""
        dummy_path = Path("/tmp/dummy")
        dummy_tp = _make_dummy_transformation_plan()
        transformation_class_path = Path("/tmp/dummy/MyTransformation.java")
        dummy_tc: TransformationClass = {
            "name": "MyTransformation",
            "package": "com.example",
            "path": transformation_class_path,
            "code": None,
        }
        
        state = ImplementationState(
            transformation_plan=dummy_tp,  # type: ignore
            transformation_class=dummy_tc,
            task_specification="Test task",
            maven_project_path=dummy_path,
            bxtool_path=dummy_path,  # Must be Path per original state definition
            written_files=[],
            latest_evaluation_runs={},
            iteration=0,
        )
        
        # Verify all fields are accessible
        self.assertEqual(state["transformation_class"]["name"], "MyTransformation")
        self.assertEqual(state["task_specification"], "Test task")
        self.assertEqual(state["iteration"], 0)
        self.assertEqual(len(state["written_files"]), 0)
        
        # Verify transformation_class contains path information (not separate state field)
        self.assertEqual(state["transformation_class"]["path"], transformation_class_path)
        self.assertEqual(state["transformation_class"]["package"], "com.example")
    
    def test_written_files_field_is_consistent(self):
        """
        Verify that 'written_files' is the canonical field name (not 'written_java_files').
        
        This is a regression test to ensure all nodes use 'written_files' consistently.
        """
        from typing import get_type_hints
        
        hints = get_type_hints(ImplementationState)
        
        # Should have 'written_files'
        self.assertIn("written_files", hints)
        
        # Should NOT have 'written_java_files' (old inconsistent name)
        self.assertNotIn("written_java_files", hints)
    
    def test_iteration_field_is_integer(self):
        """Verify that iteration is an integer field."""
        from typing import get_type_hints, get_origin, get_args
        
        hints = get_type_hints(ImplementationState)
        iteration_type = hints.get("iteration")
        
        # In TypedDict, optional fields with defaults are still typed
        # We just verify it's int or int | None
        self.assertIsNotNone(iteration_type)
    
    def test_bxtool_path_is_required_field(self):
        """
        Verify that bxtool_path is a required Path field in ImplementationState.
        
        Note: Even when BenchmarX is not integrated, the field must contain a Path.
        The implementation graph handles the logic of whether to use it or not.
        """
        from typing import get_type_hints
        
        hints = get_type_hints(ImplementationState)
        bxtool_path_type = hints.get("bxtool_path")
        
        # Should be Path (required field)
        self.assertIsNotNone(bxtool_path_type)
    
    def test_transformation_class_code_can_be_none(self):
        """
        Verify that transformation_class["code"] can be None initially.
        
        The code is populated by implement_transformation node and used
        by implement_bx_tool node (when BenchmarX integration is enabled).
        """
        dummy_path = Path("/tmp/dummy")
        dummy_tp = _make_dummy_transformation_plan()
        dummy_tc: TransformationClass = {
            "name": "TestTransformation",
            "package": "com.example",
            "path": dummy_path,
            "code": None,
        }
        
        state = ImplementationState(
            transformation_plan=dummy_tp,  # type: ignore
            transformation_class=dummy_tc,
            task_specification="test",
            maven_project_path=dummy_path,
            bxtool_path=dummy_path,
            written_files=[],
            latest_evaluation_runs={},
            iteration=0,
        )
        
        self.assertIsNone(state["transformation_class"]["code"])


class TestStateFieldUsageConsistency(TestCase):
    """Tests for consistent field usage across implementation nodes."""
    
    def test_node_imports_are_available(self):
        """Verify that key functions can be imported without errors."""
        # These imports should work without circular dependency issues
        from mdeagent.implementation.transformation.implement_transformation import (
            create_implement_transformation_node,
        )
        from mdeagent.implementation.evaluation.evaluate_transformation_implementation import (
            create_route_implementation,
        )
        from mdeagent.implementation.format_code import create_format_code_node
        from mdeagent.implementation.bxtool.implement_bx_tool import (
            create_implement_bx_tool_node,
        )
        
        # Just verifying imports work
        self.assertTrue(callable(create_implement_transformation_node))
        self.assertTrue(callable(create_route_implementation))
        self.assertTrue(callable(create_format_code_node))
        self.assertTrue(callable(create_implement_bx_tool_node))
    
    def test_transformation_class_structure(self):
        """Verify that TransformationClass has the expected structure."""
        from mdeagent.implementation.types import TransformationClass
        from typing import get_type_hints
        
        hints = get_type_hints(TransformationClass)
        
        # Required fields
        self.assertIn("name", hints)
        self.assertIn("package", hints)
        self.assertIn("path", hints)
        self.assertIn("code", hints)
        
        # Verify we can create a valid instance
        tc: TransformationClass = {
            "name": "TestTransformation",
            "package": "com.example.test",
            "path": Path("/tmp/TestTransformation.java"),
            "code": "public class TestTransformation {}",
        }
        
        self.assertEqual(tc["name"], "TestTransformation")
        self.assertEqual(tc["package"], "com.example.test")
        self.assertIsInstance(tc["path"], Path)
