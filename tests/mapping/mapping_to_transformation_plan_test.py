"""Tests for the transformation_plan mapper functions."""

from unittest import TestCase

from mdeagent.comprehension.plan import SerializedTransformationPlan
from mdeagent.mapping import mde_to_transformation_plan
from mdeagent.state import MDEAgentState


class TestMDEToTransformationPlanMapper(TestCase):
    """Test cases for mde_to_transformation_plan mapping function."""

    def _make_complete_plan(self) -> SerializedTransformationPlan:
        """Helper to build a complete SerializedTransformationPlan."""
        return {
            "data": {
                "source_model_package": "test.src",
                "target_model_package": "test.tgt",
                "iteration": 1,
                "source_model_implementation": "ECore",
                "target_model_implementation": "Java",
                "transformation_direction": "forward",
                "difficulties": "Complex type mappings",
                "implementation_steps": "Steps here",
            },
            "parser": {
                "type": "FileTransformationPlanParser",
                "args": {"file_path": "/tmp/plan.md"},
            },
            "template": "/tmp/templates",
        }

    def test_mapping_returns_transformation_plan(self):
        """Mapper should return dict with transformation_plan key."""
        plan = self._make_complete_plan()
        state = MDEAgentState(
            workspace_path="/tmp/ws",
            source_model_path="/tmp/source",
            target_model_path="/tmp/target",
            transformation_plan=plan,
        )

        result = mde_to_transformation_plan(state)

        self.assertIn("transformation_plan", result)
        self.assertEqual(result["transformation_plan"], plan)

    def test_mapping_without_transformation_plan(self):
        """Mapper should handle missing transformation_plan gracefully."""
        state = MDEAgentState(
            workspace_path="/tmp/ws",
            source_model_path="/tmp/source",
            target_model_path="/tmp/target",
        )

        result = mde_to_transformation_plan(state)

        self.assertIn("transformation_plan", result)
        self.assertIsNone(result["transformation_plan"])

    def test_mapping_only_includes_transformation_plan(self):
        """Mapper should only return transformation_plan, not all state."""
        plan = self._make_complete_plan()
        state = MDEAgentState(
            workspace_path="/tmp/ws",
            source_model_path="/tmp/source",
            target_model_path="/tmp/target",
            transformation_plan=plan,
            group_id="test.group",
            artifact_id="test-artifact",
            written_files=["/tmp/file1.java"],
        )

        result = mde_to_transformation_plan(state)

        self.assertEqual(len(result), 1)
        self.assertIn("transformation_plan", result)
        self.assertNotIn("workspace_path", result)
        self.assertNotIn("group_id", result)
        self.assertNotIn("written_files", result)

    def test_mapping_preserves_nested_data(self):
        """Mapper should preserve the full nested structure of the plan."""
        plan = self._make_complete_plan()
        state = MDEAgentState(
            workspace_path="/tmp/ws",
            source_model_path="/tmp/source",
            target_model_path="/tmp/target",
            transformation_plan=plan,
        )

        result = mde_to_transformation_plan(state)

        plan_result = result["transformation_plan"]
        self.assertIn("data", plan_result)
        self.assertIn("parser", plan_result)
        self.assertIn("template", plan_result)
        self.assertEqual(plan_result["data"]["source_model_implementation"], "ECore")
        self.assertEqual(
            plan_result["parser"]["type"], "FileTransformationPlanParser"
        )
