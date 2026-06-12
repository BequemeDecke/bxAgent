import logging
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from jinja2 import Environment, FileSystemLoader

from bxagent.tools.transformation.bxtool import BxToolForEMF, BxToolTemplateResolver


class TestBxToolTemplateResolver(TestCase):
    @patch("jinja2.Environment.get_template")
    def setUp(self, mock_get_template) -> None:
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
                    "set_configuration": "setConfiguration();",
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
                "registry": {
                    "import_path": "com.example.source.Registry",
                    "class_name": "SourceRegistry",
                    "instance_name": "sourceRegistryInstance",
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
                "registry": {
                    "import_path": "com.example.target.Registry",
                    "class_name": "TargetRegistry",
                    "instance_name": "targetRegistryInstance",
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
            "public class TransformationImplementation extends BXToolForEMF<SourceRegistry, TargetRegistry, Decisions> {",
            rendered_content,
        )
        self.assertIn("private TargetRegistry targetRegistryInstance;", rendered_content)
        self.assertIn("public TransformationImplementation() {", rendered_content)
        self.assertIn(
            "super(new SourceComparator(), new TargetComparator());",
            rendered_content,
        )
        self.assertIn(
            "source.getContents().add(sourceRegistryInstance);", rendered_content
        )
        self.assertIn(
            "target.getContents().add(targetRegistryInstance);", rendered_content
        )
        self.assertIn("initiateDialogue();", rendered_content)
        self.assertIn(
            "public void performAndPropagateEdit(Supplier<IEdit<SourceRegistry>> sourceEdit,",
            rendered_content,
        )
