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
    TransformationPlanData,
    FileTransformationPlanParser,
)


class MockedTransformationPlanParser(TransformationPlanParser):
    def __init__(
        self,
        mocked_save: Mock,
        fake_data: Dict[str, str] = {},
        fail_parsing: bool = False,
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

        self.fake_data = {
            "source_model_package": "source_package",
            "target_model_package": "target_package",
            "iteration": "1",
            "source_model_implementation": "source implementation",
            "target_model_implementation": "target implementation",
            "transformation_direction": "source to target",
            "difficulties": "some difficulties",
            "implementation_steps": "some steps",
        }

    @patch("jinja2.Environment.get_template")
    def test_transformation_plan__create_plan_existing_file(self, mock_get_template):
        """
        If the file already exists, it should not override the existing content and parse the information.
        """
        mock_get_template.return_value = Template("")

        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
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
            fake_data=self.fake_data,
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
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_source_package = "updated_source_package"
        updated_target_package = "updated_target_package"

        plan.update_package_information(
            source_model_package=updated_source_package,
            target_model_package=updated_target_package,
        )
        self.assertEqual(plan.data["source_model_package"], updated_source_package)
        self.assertEqual(plan.data["target_model_package"], updated_target_package)
        mocked_save_function.assert_called_with(str(plan))

    def test_transformation_plan__update_iteration(self):
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_iteration = 2
        plan.update_iteration(updated_iteration)
        self.assertEqual(plan.data["iteration"], updated_iteration)
        mocked_save_function.assert_called_with(str(plan))

    def test_transformation_plan__update_model_implementation(self):
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_source_implementation = "updated source implementation"
        updated_target_implementation = "updated target implementation"

        plan.update_model_implementation(
            source_model_implementation=updated_source_implementation,
            target_model_implementation=updated_target_implementation,
        )
        self.assertEqual(
            plan.data["source_model_implementation"], updated_source_implementation
        )
        self.assertEqual(
            plan.data["target_model_implementation"], updated_target_implementation
        )
        mocked_save_function.assert_called_with(str(plan))

    def test_transformation_plan__update_transformation_direction(self):
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_transformation_direction = "updated transformation direction"
        plan.update_transformation_direction(updated_transformation_direction)
        self.assertEqual(
            plan.data["transformation_direction"], updated_transformation_direction
        )
        mocked_save_function.assert_called_with(str(plan))

    def test_transformation_plan__update_transformation_difficulties(self):
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_difficulties = "updated difficulties"
        plan.update_transformation_difficulties(updated_difficulties)
        self.assertEqual(plan.data["difficulties"], updated_difficulties)
        mocked_save_function.assert_called_with(str(plan))

    def test_transformation_plan__update_implementation_steps(self):
        mocked_save_function = Mock(spec=Callable)
        mocked_parser = MockedTransformationPlanParser(
            fake_data=self.fake_data,
            mocked_save=mocked_save_function,
        )
        plan = TransformationPlan(mocked_parser)

        updated_steps = "updated steps"
        plan.update_implementation_steps(updated_steps)
        self.assertEqual(plan.data["implementation_steps"], updated_steps)
        mocked_save_function.assert_called_with(str(plan))


class TestFileTransformationPlanParser(TestCase):
    @patch("pathlib.Path.write_text")
    def test_file_transformation_plan_parser_save__writes_to_file(
        self, mocked_write_text
    ):
        mocked_write_text.return_value = None  # write_text does not return anything
        test_file_path = Path("test_transformation_plan.md")
        parser = FileTransformationPlanParser(file_path=test_file_path)
        test_data = "Test transformation plan content"
        parser.save(test_data)
        self.assertTrue(mocked_write_text.called)

    @patch("pathlib.Path.exists")
    def test_file_transformation_plan_parser__file_not_found(self, mock_exists):
        mock_exists.return_value = False
        parser = FileTransformationPlanParser(file_path=Path("non_existent_file.md"))
        with self.assertRaises(FileNotFoundError):
            parser.parse()

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_file_transformation_plan_parser__file_empty(
        self, mock_read_text, mock_exists
    ):
        mock_exists.return_value = True
        mock_read_text.return_value = ""
        parser = FileTransformationPlanParser(file_path=Path("empty_file.md"))
        with self.assertRaises(
            ValueError, msg="The transformation plan file is empty."
        ):
            parser.parse()

    def test_file_transformation_plan_parser_parse_header__invalid_format(self):
        parser = FileTransformationPlanParser(file_path=Path("invalid_file.md"))
        with self.assertRaises(
            ValueError, msg="Failed to parse transformation plan data."
        ):
            parser._parse_header("Invalid header format without the expected fields")

    def test_file_transformation_plan_parser_parse_header__valid_format(self):
        parser = FileTransformationPlanParser(file_path=Path("valid_file.md"))
        header_content = """
        ---
        source_model_package: test_source_package
        target_model_package: test_target_package
        iteration: 3
        ---
        """
        parsed_header = parser._parse_header(header_content)
        self.assertEqual(parsed_header["source_model_package"], "test_source_package")
        self.assertEqual(parsed_header["target_model_package"], "test_target_package")
        self.assertEqual(parsed_header["iteration"], 3)

    def test_file_transformation_plan_parser_parse_model_implementation__invalid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("invalid_file.md"))
        with self.assertRaises(ValueError, msg="Failed to parse model implementation."):
            parser._parse_model_implementation(
                "Invalid content without the expected markers", "source"
            )

        with self.assertRaises(ValueError, msg="Failed to parse model implementation."):
            parser._parse_model_implementation(
                "Invalid content without the expected markers", "target"
            )

    def test_file_transformation_plan_parser_parse_model_implementation__valid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("valid_file.md"))
        source_model_implementation_text = """
        This is the source model implementation.
        """

        content = f"""
        --- BEGIN SOURCE MODEL ---
        {source_model_implementation_text}
        --- END SOURCE MODEL ---
        """
        source_model_implementation = parser._parse_model_implementation(
            content, "source"
        )
        self.assertEqual(
            source_model_implementation, source_model_implementation_text.strip()
        )

        target_model_implementation = """
        This is the multiline 
        target model implementation.
        Normally this would be some java code block, but for testing purposes we keep it simple.
        """
        content = f"""
        --- BEGIN TARGET MODEL ---
        {target_model_implementation}
        --- END TARGET MODEL ---
        """
        target_model_implementation = parser._parse_model_implementation(
            content, "target"
        )
        self.assertEqual(
            target_model_implementation, target_model_implementation.strip()
        )

    def test_file_transformation_plan_parser_parse_transformation_direction__invalid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("invalid_file.md"))
        with self.assertRaises(
            ValueError, msg="Failed to parse transformation direction."
        ):
            parser._parse_transformation_direction(
                "Invalid content without the expected markers"
            )

    def test_file_transformation_plan_parser_parse_transformation_direction__valid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("valid_file.md"))
        transformation_direction_text = "source to target"
        content = f"""
        --- BEGIN TRANSFORMATION DIRECTION ---
        {transformation_direction_text}
        --- END TRANSFORMATION DIRECTION ---
        """
        transformation_direction = parser._parse_transformation_direction(content)
        self.assertEqual(
            transformation_direction, transformation_direction_text.strip()
        )

    def test_file_transformation_plan_parser_parse_difficulties__invalid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("invalid_file.md"))
        with self.assertRaises(
            ValueError, msg="Failed to parse transformation difficulties."
        ):
            parser._parse_difficulties("Invalid content without the expected markers")

    def test_file_transformation_plan_parser_parse_difficulties__valid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("valid_file.md"))
        difficulties_text = "Some difficulties with the transformation."
        content = f"""
        --- BEGIN DIFFICULTIES ---
        {difficulties_text}
        --- END DIFFICULTIES ---
        """
        difficulties = parser._parse_difficulties(content)
        self.assertEqual(difficulties, difficulties_text.strip())

    def test_file_transformation_plan_parser_parse_implementation_steps__invalid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("invalid_file.md"))
        with self.assertRaises(ValueError, msg="Failed to parse implementation steps."):
            parser._parse_implementation_steps(
                "Invalid content without the expected markers"
            )

    def test_file_transformation_plan_parser_parse_implementation_steps__valid_format(
        self,
    ):
        parser = FileTransformationPlanParser(file_path=Path("valid_file.md"))
        steps_text = "1. First step\n2. Second step\n3. Third step"
        content = f"""
        --- BEGIN IMPLEMENTATION STEPS ---
        {steps_text}
        --- END IMPLEMENTATION STEPS ---
        """
        implementation_steps = parser._parse_implementation_steps(content)
        self.assertEqual(implementation_steps, steps_text.strip())

simpson_model = """
```java
class Simpson {
    private String movie;
}
```
"""
family_member_model = """
```java
class FamilyMember {
    private String name;
}
```
"""

class TestTransformationPlanFileIntegration(TestCase):
    def setUp(self):
        self.maxDiff = None  # To see the full diff in case of assertion failures
        
        self.input_file = Path(
            "tests/tools/transformation/files/EXISTED_TRANSFORMATION.md"
        )
        self.expected_file = Path(
            "tests/tools/transformation/files/EXPECTED_TRANSFORMATION.md"
        )

        self.assertTrue(
            self.input_file.exists(),
            f"Input file does not exist at {self.input_file.resolve()}",
        )
        self.assertTrue(
            self.expected_file.exists(),
            f"Expected file does not exist at {self.expected_file.resolve()}",
        )

        self.parser = FileTransformationPlanParser(file_path=self.input_file)
        self.plan = TransformationPlan(self.parser)

    @patch("pathlib.Path.write_text")
    def test_transformation_plan_file_integration__round_trip_of_usage(self, mock_write_text):
        mock_write_text.return_value = None  # write_text does not return anything
        
        # Update the package information and check if the file is updated accordingly.
        self.plan.update_package_information(
            source_model_package="de.hof-university.models.simpson",
            target_model_package="de.hof-university.models.family",
        )
        self.assertTrue(mock_write_text.called)

        # Update the iteration and check if the file is updated accordingly.
        self.plan.update_iteration(2)
        self.assertTrue(mock_write_text.called)

        # Update the model implementations and check if the file is updated accordingly.
       

        self.plan.update_model_implementation(
            source_model_implementation=simpson_model.strip(),
            target_model_implementation=family_member_model.strip(),
        )
        self.assertTrue(mock_write_text.called)

        # Update the transformation direction and check if the file is updated accordingly.
        self.plan.update_transformation_direction("Do everything!")
        self.assertTrue(mock_write_text.called)

        # Update the difficulties and check if the file is updated accordingly.
        self.plan.update_transformation_difficulties(
            "No difficulties. Looks pretty simple!"
        )

        # Update the implementation steps and check if the file is updated accordingly.
        self.plan.update_implementation_steps("1. Start IMPLEMENTING!!!")

        expected_content = self.expected_file.read_text()
        actual_content = str(self.plan)
        self.assertEqual(
            expected_content.strip(),
            actual_content.strip(),
            "The content of the transformation plan file does not match the expected content after updates.",
        )
