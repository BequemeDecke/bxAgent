"""Tests for the TransformationPlanEvaluation evaluator."""

import asyncio
import unittest

from mdeagent.comprehension.plan import SerializedTransformationPlan
from mdeagent.evaluation.implementations.transformation_plan import (
    TransformationPlanEvaluation,
    TransformationPlanSchema,
)


def _make_full_plan(
    source_model_implementation="",
    target_model_implementation="",
    transformation_direction="",
    difficulties="",
    implementation_steps="",
) -> SerializedTransformationPlan:
    """Build a complete SerializedTransformationPlan suitable for the evaluator."""
    return {
        "data": {
            "source_model_package": "test.src",
            "target_model_package": "test.tgt",
            "iteration": 0,
            "source_model_implementation": source_model_implementation,
            "target_model_implementation": target_model_implementation,
            "transformation_direction": transformation_direction,
            "difficulties": difficulties,
            "implementation_steps": implementation_steps,
        },
        "parser": {
            "type": "FileTransformationPlanParser",
            "args": {"file_path": "/tmp/plan.md"},
        },
        "template": "/tmp/templates",
    }


class TestTransformationPlanSchema(unittest.TestCase):
    """Test the schema validation for transformation plan parameters."""

    def test_schema_validates_complete_plan(self):
        """Schema should accept a dict with transformation_plan key."""
        data = {
            "transformation_plan": _make_full_plan(
                source_model_implementation="ECore",
                target_model_implementation="Java",
                transformation_direction="forward",
                difficulties="Complex mappings",
                implementation_steps="Steps",
            )
        }
        # Should not raise
        result = TransformationPlanSchema.model_validate(data)
        self.assertIsNotNone(result)

    def test_schema_rejects_missing_plan(self):
        """Schema should reject input missing the transformation_plan key."""
        data = {"other_key": "value"}
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for missing plan."
        ):
            TransformationPlanSchema.model_validate(data)

    def test_schema_allows_extra_keys(self):
        """Schema should allow extra keys beyond transformation_plan."""
        data = {
            "transformation_plan": _make_full_plan(),
            "extra": "ignored",
        }
        result = TransformationPlanSchema.model_validate(data)
        self.assertIsNotNone(result)


class TestTransformationPlanEvaluation(unittest.TestCase):
    """Test the TransformationPlanEvaluation logic."""

    def test_evaluation_empty_plan_none(self):
        """When plan is None, should return a single failure result."""
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=None))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 0)
        self.assertFalse(results[0].metadata.get("success", True))
        self.assertTrue(results[0].metadata.get("include_in_report"))
        self.assertIn("No transformation plan", results[0].content)

    def test_evaluation_empty_plan_empty_dict(self):
        """When plan is empty dict, should return a single failure result."""
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan={}))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(errors), 0)
        self.assertFalse(results[0].metadata.get("success", True))

    def test_evaluation_complete_plan(self):
        """When all fields are populated, all results should be successful."""
        plan = _make_full_plan(
            source_model_implementation="ECore",
            target_model_implementation="Java",
            transformation_direction="forward",
            difficulties="Complex type mappings",
            implementation_steps="Define mapping, generate code",
        )
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=plan))

        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        for result in results:
            self.assertTrue(result.metadata.get("success", False))

        # Verify all are excluded from report
        for result in results:
            self.assertFalse(result.metadata.get("include_in_report", True))

    def test_evaluation_partial_plan(self):
        """When some fields are missing, those should fail while others pass."""
        plan = _make_full_plan(
            source_model_implementation="ECore",
            target_model_implementation="",
            transformation_direction="forward",
            difficulties="",
            implementation_steps="Steps",
        )
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=plan))

        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)

        success_count = sum(1 for r in results if r.metadata.get("success"))
        self.assertEqual(success_count, 3)

        fail_count = sum(1 for r in results if not r.metadata.get("success"))
        self.assertEqual(fail_count, 2)

        # Verify only failures are included in report
        report_items = [r for r in results if r.metadata.get("include_in_report")]
        self.assertEqual(len(report_items), 2)

        # Verify the failing items are target_model and difficulties
        fail_contents = {r.content for r in report_items}
        self.assertTrue(any("Target model" in c for c in fail_contents))
        self.assertTrue(any("difficulties" in c.lower() for c in fail_contents))

    def test_evaluation_whitespace_only_fields(self):
        """Whitespace-only fields should be treated as empty."""
        plan = _make_full_plan(
            source_model_implementation="   ",
            target_model_implementation="Java",
            transformation_direction="\t\n",
            difficulties="  ",
            implementation_steps="  Steps  ",
        )
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=plan))

        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)

        # Only target_model_implementation and implementation_steps should pass
        success_count = sum(1 for r in results if r.metadata.get("success"))
        self.assertEqual(success_count, 2)

    def test_evaluation_checks_all_required_fields(self):
        """Verify each required field gets its own result entry."""
        plan = _make_full_plan(
            source_model_implementation="Src",
            target_model_implementation="Tgt",
            transformation_direction="dir",
            difficulties="diff",
            implementation_steps="steps",
        )
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=plan))

        self.assertEqual(len(results), 5)

        contents = {r.content for r in results}
        self.assertTrue(any("Source model" in c for c in contents))
        self.assertTrue(any("Target model" in c for c in contents))
        self.assertTrue(any("direction" in c.lower() for c in contents))
        self.assertTrue(any("difficulties" in c.lower() for c in contents))
        self.assertTrue(any("step" in c.lower() for c in contents))

    def test_evaluation_no_errors(self):
        """TransformationPlanEvaluation should never produce errors."""
        plan = _make_full_plan()
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=plan))

        self.assertEqual(len(errors), 0)

    def test_setup_is_noop(self):
        """Setup should be async and do nothing."""
        evaluator = TransformationPlanEvaluation()

        # Should not raise
        asyncio.run(evaluator.setup())

    def test_evaluation_handles_flat_dict(self):
        """If a flat dict is passed (no 'data' key), handle it gracefully."""
        flat_plan = {
            "source_model_implementation": "ECore",
            "target_model_implementation": "Java",
            "transformation_direction": "forward",
            "difficulties": "Some difficulties",
            "implementation_steps": "Steps",
        }
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=flat_plan))

        # Should have 5 results, all passing
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        success_count = sum(1 for r in results if r.metadata.get("success"))
        self.assertEqual(success_count, 5)

    def test_evaluation_handles_nested_data_key(self):
        """If plan has nested 'data' key, extract data correctly."""
        plan = _make_full_plan(
            source_model_implementation="Src",
            target_model_implementation="Tgt",
            transformation_direction="fwd",
            difficulties="diff",
            implementation_steps="steps",
        )
        # Simulate the exact structure returned by tp.to_dict()
        nested_plan = {
            "data": plan["data"],
            "parser": plan["parser"],
            "template": plan["template"],
        }
        evaluator = TransformationPlanEvaluation()

        results, errors = asyncio.run(evaluator.run(transformation_plan=nested_plan))

        # Should have 5 results, all passing
        self.assertEqual(len(results), 5)
        self.assertEqual(len(errors), 0)
        success_count = sum(1 for r in results if r.metadata.get("success"))
        self.assertEqual(success_count, 5)
