import logging
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from jinja2 import Environment, FileSystemLoader

from mdeagent.implementation.bxtool import BxToolForEMF, BxToolTemplateResolver


class TestBxToolTemplateResolver(TestCase):
    @patch("jinja2.Environment.get_template")
    def setUp(self, mock_get_template: Mock) -> None:
        self.maxDiff = None  # type: ignore
        # It references the correct template, but due to the testing environment, there might be issues with loading the template. We can mock the template loading to ensure that the tests run without issues.
        path = Path("templates/bxtool.jinja")
        logging.debug(f"Path: {path.resolve()}, Exists: {path.exists()}")
        mock_get_template.return_value = Environment(
            loader=FileSystemLoader("templates")
        ).get_template("bxtool.jinja")

        self.fake_data = {
            "transformation_package": "com.example.transformation",
            "transformation_implementation": {
                "import_path": "com.example.transformation.TransformationImplementation",
                "class_name": "TransformationImplementation",
                "instance_name": "transformationImplementationInstance",
                "decisions": {
                    "import_path": "com.example.transformation.Decisions",
                },
                "initiation_dialogue": {
                    "set_configuration": "setConfigurator();",
                    "initiate_dialogue": "initiateDialogue();",
                },
                "perform_and_propagate_target_edit": "performAndPropagateTargetEdit();",
                "perform_and_propagate_source_edit": "performAndPropagateSourceEdit();",
                "perform_and_propagate_concurrent_edit": "performAndPropagateConcurrentEdit();",
            },
            "source_model": {
                "name": "SourceModel",
                "factory": {
                    "import_path": "com.example.source.Factory",
                    "class_name": "SourceFactory",
                    "instance_name": "sourceFactoryInstance",
                },
                "register": {
                    "import_path": "com.example.source.Register",
                    "class_name": "SourceRegister",
                    "instance_name": "sourceRegisterInstance",
                },
                "comparator": {
                    "import_path": "com.example.source.Comparator",
                    "class_name": "SourceComparator",
                    "instance_name": "sourceComparatorInstance",
                },
            },
            "target_model": {
                "name": "TargetModel",
                "factory": {
                    "import_path": "com.example.target.Factory",
                    "class_name": "TargetFactory",
                    "instance_name": "targetFactoryInstance",
                },
                "register": {
                    "import_path": "com.example.target.Register",
                    "class_name": "TargetRegister",
                    "instance_name": "targetRegisterInstance",
                },
                "comparator": {
                    "import_path": "com.example.target.Comparator",
                    "class_name": "TargetComparator",
                    "instance_name": "targetComparatorInstance",
                },
            },
            "additional_imports": [
                "com.example.additional.Import1",
                "com.example.additional.Import2",
            ],
        }

    def test_bxtool_template_resolver__render_template(self):
        resolver = BxToolTemplateResolver()
        bx_tool_for_emf = BxToolForEMF.model_construct(**self.fake_data)
        rendered_content = resolver.render_template(bx_tool_for_emf)
        self.assertIn("package com.example.transformation;", rendered_content)
        self.assertIn("import org.benchmarx.config.Configurator;", rendered_content)
        self.assertIn("import com.example.additional.Import1;", rendered_content)
        self.assertIn(
            "public class TransformationImplementation extends BXToolForEMF<SourceRegister, TargetRegister, Decisions> {",
            rendered_content,
        )
        self.assertIn(
            "private TargetRegister targetRegisterInstance;", rendered_content
        )
        self.assertIn("public TransformationImplementation() {", rendered_content)
        self.assertIn(
            "super(new SourceComparator(), new TargetComparator());",
            rendered_content,
        )
        self.assertIn(
            "source.getContents().add(sourceRegisterInstance);", rendered_content
        )
        self.assertIn(
            "target.getContents().add(targetRegisterInstance);", rendered_content
        )
        self.assertIn("initiateDialogue();", rendered_content)
        self.assertIn(
            "public void performAndPropagateEdit(Supplier<IEdit<SourceRegister>> sourceEdit,",
            rendered_content,
        )

    def test_bxtool_template_resolver__complete_file(self):
        expected_file_path = Path(
            "tests/comprehension/files/TransformationImplementation.java"
        )
        self.assertTrue(
            expected_file_path.exists(),
            f"Expected file {expected_file_path} does not exist.",
        )
        expected_file_content = expected_file_path.read_text().strip()

        resolver = BxToolTemplateResolver()
        bx_tool_for_emf = BxToolForEMF.model_construct(**self.fake_data)
        rendered_content = resolver.render_template(bx_tool_for_emf).strip()

        self.assertEqual(expected_file_content, rendered_content)
