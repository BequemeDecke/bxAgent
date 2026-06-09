"""
The TRANSFORMATION.md file is a crucial component and encapsulates the information about the transformation.

The file now should have a template structure that lets the agent simply fill in or retrieve the necessary information for the transformation.
"""

import logging

from unittest import TestCase
from unittest.mock import patch, Mock
from typing import Dict, Callable
from pathlib import Path
from jinja2 import Template, Environment, FileSystemLoader

from bxagent.tools.transformation.plan import (
    TransformationPlan,
    TransformationPlanParser,
)


class MockedTransformationPlanParser(TransformationPlanParser):
    def __init__(
        self, mocked_save: Mock, fake_data: Dict[str, str], fail_parsing: bool = False
    ):
        self.fake_data = fake_data
        self.mocked_save = mocked_save
        self.fail_parsing = fail_parsing

    def parse(self):
        if self.fail_parsing:
            raise ValueError("Failed to parse transformation plan")
        return self.fake_data

    def save(self, data: str) -> None:
        self.mocked_save(data)


class TestTransformationPlan(TestCase):
    @patch("jinja2.Environment.get_template")
    def setUp(self, mock_get_template) -> None:
        # It references the correct template, but due to the testing environment, there might be issues with loading the template. We can mock the template loading to ensure that the tests run without issues.
        path = Path("templates/transformation_plan.jinja")
        logging.debug(f"Path: {path.resolve()}, Exists: {path.exists()}")
        mock_get_template.return_value = Environment(
            loader=FileSystemLoader("templates")
        ).get_template("transformation_plan.jinja")

    @patch("jinja2.Environment.get_template")
    def test_transformation_plan__create_plan_existing_file(self, mock_get_template):
        """
        If the file already exists, it should not override the existing content and parse the information.
        """
        mock_get_template.return_value = Template("")

        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data={
                "source_model_package": "source_package",
                "target_model_package": "target_package",
                "iteration": "1",
                "source_model_implementation": "source implementation",
                "target_model_implementation": "target implementation",
                "transformation_direction": "source to target",
                "difficulties": "some difficulties",
                "implementation_steps": "some steps",
            },
            mocked_save=mocked_save_function,
        )

        plan = TransformationPlan(mocked_parser)
        self.assertEqual(plan.data["source_model_package"], "source_package")
        self.assertEqual(plan.data["target_model_package"], "target_package")
        self.assertEqual(plan.data["iteration"], "1")
        self.assertEqual(
            plan.data["source_model_implementation"], "source implementation"
        )
        self.assertEqual(
            plan.data["target_model_implementation"], "target implementation"
        )
        self.assertEqual(plan.data["transformation_direction"], "source to target")
        self.assertEqual(plan.data["difficulties"], "some difficulties")
        self.assertEqual(plan.data["implementation_steps"], "some steps")

    @patch("jinja2.Environment.get_template")
    def test_transformation_plan__create_plan_parsing_fails(self, mock_get_template):
        """
        If parsing the existing file fails, it should handle the error and initialize with empty values.
        """
        mock_get_template.return_value = Template("")

        mocked_parser = MockedTransformationPlanParser(
            mocked_save=Mock(spec=Callable),
            fail_parsing=True,
        )

        plan = TransformationPlan(mocked_parser)
        self.assertEqual(plan.data["source_model_package"], "")
        self.assertEqual(plan.data["target_model_package"], "")
        self.assertEqual(plan.data["iteration"], 0)
        self.assertEqual(plan.data["source_model_implementation"], "")
        self.assertEqual(plan.data["target_model_implementation"], "")
        self.assertEqual(plan.data["transformation_direction"], "")
        self.assertEqual(plan.data["difficulties"], "")
        self.assertEqual(plan.data["implementation_steps"], "")

    def test_transformation_plan__stringify_plan(self):
        mocked_parser = MockedTransformationPlanParser(
            fake_data={
                "source_model_package": "source_package",
                "target_model_package": "target_package",
                "iteration": "1",
                "source_model_implementation": "source implementation",
                "target_model_implementation": "target implementation",
                "transformation_direction": "source to target",
                "difficulties": "some difficulties",
                "implementation_steps": "some steps",
            },
            mocked_save=Mock(spec=Callable),
        )
        plan = TransformationPlan(mocked_parser)
        self.assertIn("source_package", str(plan))
        self.assertIn("target_package", str(plan))
        self.assertIn("## 1. Model Implementations", str(plan))
        self.assertIn("source implementation", str(plan))
        self.assertIn("target implementation", str(plan))
        self.assertIn(
            "## 2. Transformation Direction", str(plan)
        )  # Some probes to ensure that the template is used
        self.assertIn("source to target", str(plan))
        self.assertIn("some difficulties", str(plan))
        self.assertIn("some steps", str(plan))

    def test_transformation_plan__update_package_information(self):
        self.assertTrue(False)

    def test_transformation_plan__update_iteration(self):
        self.assertTrue(False)

    def test_transformation_plan__update_model_implementation(self):
        self.assertTrue(False)

    def test_transformation_plan__update_transformation_direction(self):
        self.assertTrue(False)

    def test_transformation_plan__update_transformation_difficulties(self):
        self.assertTrue(False)

    def test_transformation_plan__update_implementation_steps(self):
        self.assertTrue(False)
