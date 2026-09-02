from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain.chat_models import BaseChatModel

from mdeagent.implementation.generator import (
    TransformationClassSpec,
    TransformationClassTemplateResolver,
    create_generate_transformation_node,
)
from mdeagent.implementation.state import ImplementationState


class TestTransformationGeneration(TestCase):
    def setUp(self):
        self.fake_data = {
            "package_name": "com.example.transformation",
            "class_name": "MyTransformation",
            "source_type": "SourceModel",
            "target_type": "TargetModel",
            "decision_type": "TransformationDecisions",
            "fields": [{"type": "String", "name": "label"}],
            "constructor": {
                "parameters": "String label",
                "assignments": [{"target": "label", "value": "label"}],
            },
            "forward_body": "System.out.println(source);",
            "backward_body": "System.out.println(target);",
            "synch_body": 'System.out.println("synced");',
            "transform_source_to_target_body": "forward(source, target, decisions);",
            "transform_target_to_source_body": "backward(target, source, decisions);",
        }

        mocked_llm = Mock(spec=BaseChatModel)
        mocked_llm_structured_output = Mock(spec=BaseChatModel)
        mocked_llm.with_structured_output.return_value = mocked_llm_structured_output
        mocked_llm_structured_output.invoke.return_value = TransformationClassSpec(
            **self.fake_data
        )

        self.generate_transformation = create_generate_transformation_node(
            mocked_llm, Path("/tmp/workspace")
        )

    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_generate_transformation__write_template_output(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
    ):
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        state = ImplementationState(
            transformation_md=None,
            task_specification="Implement the transformation class",
            written_java_files=[],
            bxtool_path=Path("/tmp/workspace"),
            transformation_implementation="public class MyTransformation { }",
            latest_evaluation_results={},
            implementation_iteration=1,
        )

        new_state = self.generate_transformation(state)

        self.assertEqual(len(new_state["written_java_files"]), 1)
        self.assertEqual(
            new_state["written_java_files"][0],
            Path("/tmp/workspace/MyTransformation.java"),
        )
        mock_get_raw_template.assert_called_once()
        mock_render_template.assert_called_once_with(
            TransformationClassSpec(**self.fake_data)
        )

    def test_transformation_class_template_resolver__renders_emf_interface(self):
        resolver = TransformationClassTemplateResolver()
        rendered = resolver.render_template(TransformationClassSpec(**self.fake_data))

        self.assertIn(
            "implements AgentTransformationForEMF<",
            rendered,
        )
        self.assertIn("SourceModel", rendered)
        self.assertIn("TargetModel", rendered)
        self.assertIn("TransformationDecisions", rendered)
        self.assertIn("public void forward(SourceModel source", rendered)
