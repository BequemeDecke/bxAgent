"""
Test cases for the implement_transformation node.

This node is part of the coding agent built with langgraph and is responsible for implementing the transformation logic based on the provided specifications and requirements.

It should read the `TRANSFORMATION.md` file for necessary information and use that to generate the appropriate java code for the transformation.

This component uses a few llm calls which will make testing more difficult. Therefore the tests consists of two types:
1. Unit tests: These tests will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent.
"""

from unittest import TestCase
from unittest.mock import patch

from bxagent.tools.coding.implement_transformation import (
    get_transformation_plan,
    create_implement_transformation_node,
)


class TestGetTransformationPlan(TestCase):
    @patch("bxagent.tools.transformation._read_transformation_plan")
    def test_get_transformation_plan__empty_transformation_md(self, mock_read_plan):
        transformation_md = "Some content of the transformation ..."
        mock_read_plan.return_value = transformation_md

        actual = get_transformation_plan(None)
        self.assertEqual(
            actual,
            transformation_md,
            "If None is provided, read the transformation plan from the TRANSFORMATION.md file",
        )

        actual = get_transformation_plan("")
        self.assertEqual(
            actual,
            transformation_md,
            "If an empty string is provided, read the transformation plan from the TRANSFORMATION.md file",
        )

        mock_read_plan.return_value = ""
        with self.assertRaises(
            ValueError,
            msg="If the transformation plan is empty, a ValueError should be raised",
        ):
            get_transformation_plan(None)

        with self.assertRaises(
            ValueError,
            msg="If the transformation plan is empty, a ValueError should be raised",
        ):
            get_transformation_plan("")

    @patch("bxagent.tools.transformation._read_transformation_plan")
    def test_get_transformation_plan__provided_transformation_md(self, mock_read_plan):
        transformation_md = "Some content of the transformation ..."
        mock_read_plan.return_value = "This should not be returned"

        actual = get_transformation_plan(transformation_md)
        self.assertEqual(
            actual,
            transformation_md,
            "If a non-empty transformation_md is provided, it should be returned instead of reading from the TRANSFORMATION.md file",
        )


class TestImplementTransformation(TestCase):
    def test_implement_transformation__return_compiled_code(self):
        self.assertTrue(False)
