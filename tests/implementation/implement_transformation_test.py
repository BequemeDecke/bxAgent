"""
Test cases for the implement_transformation node.

This node is part of the coding agent built with langgraph and is responsible for implementing the transformation logic based on the provided specifications and requirements.

It should read the `TRANSFORMATION.md` file for necessary information and use that to generate the appropriate java code for the transformation.

This component uses a few llm calls which will make testing more difficult. Therefore the tests consists of two types:
1. Unit tests: These tests will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent.
"""

from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain.chat_models import BaseChatModel

from mdeagent.comprehension.plan import TransformationPlan, TransformationPlanParser
from mdeagent.implementation.generator import (
    TransformationClassSpec,
    TransformationClassTemplateResolver,
)
from mdeagent.implementation.implement_transformation import (
    create_implement_transformation_node,
    create_input_prompt,
)
from mdeagent.implementation.state import ImplementationState


class TestCreateInputPrompt(TestCase):
    def test_create_input_prompt__returns_correctly_formatted_prompt(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "This is the transformation plan."
        template = "public class {{ class_name }} { }"

        actual_prompt = create_input_prompt(
            task_specification, transformation_plan, template
        )

        self.assertIn("--- BEGIN TASK SPECIFICATION ---", actual_prompt)
        self.assertIn(task_specification, actual_prompt)
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", actual_prompt)
        self.assertIn(transformation_plan, actual_prompt)
        self.assertIn("--- BEGIN TEMPLATE ---", actual_prompt)
        self.assertIn(template, actual_prompt)

    def test_create_input_prompt__includes_difficulties_from_plan(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "--- BEGIN DIFFICULTIES ---\nSome difficulties here\n--- END DIFFICULTIES ---"
        template = "public class {{ class_name }} { }"

        actual_prompt = create_input_prompt(
            task_specification, transformation_plan, template
        )

        self.assertIn("--- BEGIN DIFFICULTIES ---", transformation_plan)
        self.assertIn("Some difficulties here", actual_prompt)


class TestImplementTransformation(TestCase):
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
            "synch_body": "System.out.println(\"synced\");",
            "transform_source_to_target_body": "forward(source, target, decisions);",
            "transform_target_to_source_body": "backward(target, source, decisions);",
        }

        self.mocked_llm = Mock(spec=BaseChatModel)
        self.mocked_llm_structured_output = Mock(spec=BaseChatModel)
        self.mocked_llm.with_structured_output.return_value = (
            self.mocked_llm_structured_output
        )
        self.mocked_llm_structured_output.invoke.return_value = TransformationClassSpec(
            **self.fake_data
        )

    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__writes_template_output(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
    ):
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            workspace=Path("/tmp/workspace"),
            optional_plan_factory=lambda: None,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_implementation": "",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        actual_state = implement_transformation(state)

        self.assertTrue(self.mocked_llm_structured_output.invoke.called)
        self.assertIn("written_java_files", actual_state)
        self.assertEqual(
            actual_state["written_java_files"],
            [Path("/tmp/workspace/MyTransformation.java")],
        )
        self.assertIn("transformation_implementation", actual_state)
        mock_get_raw_template.assert_called_once()
        mock_render_template.assert_called_once()

    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__appends_to_existing_files(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
    ):
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            workspace=Path("/tmp/workspace"),
            optional_plan_factory=lambda: None,
        )

        existing_file = Path("/tmp/workspace/ExistingTransformation.java")
        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [existing_file],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_implementation": "",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        actual_state = implement_transformation(state)

        self.assertIn(existing_file, actual_state["written_java_files"])
        self.assertIn(Path("/tmp/workspace/MyTransformation.java"), actual_state["written_java_files"])
        self.assertEqual(len(actual_state["written_java_files"]), 2)

    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__uses_transformation_plan(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
    ):
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        mocked_parser = Mock(spec=TransformationPlanParser)
        mocked_transformation_plan = Mock(spec=TransformationPlan)
        mocked_transformation_plan.__str__ = Mock(return_value="This is the transformation plan.")
        
        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            workspace=Path("/tmp/workspace"),
            optional_plan_factory=lambda: mocked_transformation_plan,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": mocked_transformation_plan,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_implementation": "",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        actual_state = implement_transformation(state)

        self.assertEqual(actual_state["transformation_md"], mocked_transformation_plan)
        self.assertTrue(mocked_transformation_plan.__str__.called)

    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__includes_plan_in_prompt(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
    ):
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        plan_content = "This is the transformation plan."
        mocked_parser = Mock(spec=TransformationPlanParser)
        mocked_transformation_plan = Mock(spec=TransformationPlan)
        mocked_transformation_plan.__str__ = Mock(return_value=plan_content)
        
        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            workspace=Path("/tmp/workspace"),
            optional_plan_factory=lambda: mocked_transformation_plan,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": mocked_transformation_plan,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_implementation": "",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        actual_state = implement_transformation(state)

        # Verify that the LLM was called with a prompt that includes the transformation plan
        call_args = self.mocked_llm_structured_output.invoke.call_args
        prompt = call_args.kwargs.get("input") or call_args.args[0]
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", prompt)
        self.assertIn(plan_content, prompt)


class TestTransformationClassTemplateResolver(TestCase):
    def test_transformation_class_template_resolver__renders_emf_interface(self):
        resolver = TransformationClassTemplateResolver()
        fake_data = {
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
            "synch_body": "System.out.println(\"synced\");",
            "transform_source_to_target_body": "forward(source, target, decisions);",
            "transform_target_to_source_body": "backward(target, source, decisions);",
        }
        rendered = resolver.render_template(TransformationClassSpec(**fake_data))

        self.assertIn("implements AgentTransformationForEMF<", rendered)
        self.assertIn("SourceModel", rendered)
        self.assertIn("TargetModel", rendered)
        self.assertIn("TransformationDecisions", rendered)
        self.assertIn("public void forward(SourceModel source", rendered)