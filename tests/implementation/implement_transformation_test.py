"""
Test cases for the implement_transformation node.

This node is part of the coding agent built with langgraph and is responsible for implementing the transformation logic based on the provided specifications and requirements.

It should read the `TRANSFORMATION.md` file for necessary information and use that to generate the appropriate java code for the transformation.

This component uses a few llm calls which will make testing more difficult. Therefore the tests consists of two types:
1. Unit tests: These will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent.
"""

import asyncio
from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from langchain.chat_models import BaseChatModel

from mdeagent.comprehension.plan import TransformationPlan, TransformationPlanParser
from mdeagent.implementation.generator import (
    BackwardMethodBody,
    ForwardMethodBody,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassSpec,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
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
        self.fake_metadata = {
            "package_name": "com.example.transformation",
            "source_type": "SourceModel",
            "target_type": "TargetModel",
            "decision_type": "TransformationDecisions",
            "transformation_package": "com.example",
        }

        self.fake_fields_constructor = {
            "fields": [{"type": "String", "name": "label"}],
            "constructor": {
                "parameters": "String label",
                "assignments": [{"target": "label", "value": "label"}],
            },
        }

        self.fake_forward_body = {"forward_body": "System.out.println(source);"}
        self.fake_backward_body = {"backward_body": "System.out.println(target);"}
        self.fake_synch_body = {"synch_body": 'System.out.println("synced");'}

        self.mocked_llm = Mock(spec=BaseChatModel)

        # Create separate mocks for each structured output LLM
        self.mocked_metadata_llm = Mock(spec=BaseChatModel)
        self.mocked_fields_llm = Mock(spec=BaseChatModel)
        self.mocked_forward_llm = Mock(spec=BaseChatModel)
        self.mocked_backward_llm = Mock(spec=BaseChatModel)
        self.mocked_synch_llm = Mock(spec=BaseChatModel)

        # Configure with_structured_output to return the appropriate mock based on schema
        def mock_with_structured_output(schema):
            from mdeagent.implementation.generator import (
                BackwardMethodBody,
                ForwardMethodBody,
                SynchMethodBody,
                TransformationClassMetadata,
                TransformationFieldsAndConstructor,
            )

            if schema == TransformationClassMetadata:
                return self.mocked_metadata_llm
            elif schema == TransformationFieldsAndConstructor:
                return self.mocked_fields_llm
            elif schema == ForwardMethodBody:
                return self.mocked_forward_llm
            elif schema == BackwardMethodBody:
                return self.mocked_backward_llm
            elif schema == SynchMethodBody:
                return self.mocked_synch_llm
            else:
                # Fallback for any other schema (e.g., ImplementationTransformationSpec)
                fallback_mock = Mock(spec=BaseChatModel)
                return fallback_mock

        self.mocked_llm.with_structured_output = Mock(
            side_effect=mock_with_structured_output
        )

        # Set up ainvoke mocks for each LLM
        self.mocked_metadata_llm.ainvoke = AsyncMock(
            return_value=TransformationClassMetadata(**self.fake_metadata)
        )
        self.mocked_fields_llm.ainvoke = AsyncMock(
            return_value=TransformationFieldsAndConstructor(
                **self.fake_fields_constructor
            )
        )
        self.mocked_forward_llm.ainvoke = AsyncMock(
            return_value=ForwardMethodBody(**self.fake_forward_body)
        )
        self.mocked_backward_llm.ainvoke = AsyncMock(
            return_value=BackwardMethodBody(**self.fake_backward_body)
        )
        self.mocked_synch_llm.ainvoke = AsyncMock(
            return_value=SynchMethodBody(**self.fake_synch_body)
        )

    @patch("pathlib.Path.touch")
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
        mock_touch: Mock,
    ):
        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        actual_state = asyncio.run(implement_transformation(state))

        # Verify all 5 LLMs were called (metadata + 4 parallel calls)
        self.assertTrue(self.mocked_metadata_llm.ainvoke.called)
        self.assertTrue(self.mocked_fields_llm.ainvoke.called)
        self.assertTrue(self.mocked_forward_llm.ainvoke.called)
        self.assertTrue(self.mocked_backward_llm.ainvoke.called)
        self.assertTrue(self.mocked_synch_llm.ainvoke.called)

        self.assertIn("written_java_files", actual_state)
        self.assertEqual(
            actual_state["written_java_files"],
            [Path("/tmp/workspace/MyTransformation.java")],
        )
        self.assertIn("transformation_implementation", actual_state)
        # The iteration counter is advanced by the `evaluate_implementation`
        # node, NOT by this work node (see `agent.py`).
        self.assertNotIn("iteration", actual_state)
        mock_get_raw_template.assert_called_once()
        mock_render_template.assert_called_once()

    @patch("pathlib.Path.touch")
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
        mock_touch: Mock,
    ):
        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        existing_file = Path("/tmp/workspace/ExistingTransformation.java")
        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [existing_file],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        actual_state = asyncio.run(implement_transformation(state))

        self.assertIn(existing_file, actual_state["written_java_files"])
        self.assertIn(
            Path("/tmp/workspace/MyTransformation.java"),
            actual_state["written_java_files"],
        )
        self.assertEqual(len(actual_state["written_java_files"]), 2)
        self.assertNotIn("iteration", actual_state)

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
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
        mock_write_text: Mock,
        mock_touch: Mock,
    ):
        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        mocked_parser = Mock(spec=TransformationPlanParser)
        mocked_transformation_plan = Mock(spec=TransformationPlan)
        mocked_transformation_plan.__str__ = Mock(
            return_value="This is the transformation plan."
        )

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: mocked_transformation_plan,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": mocked_transformation_plan,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        actual_state = asyncio.run(implement_transformation(state))

        self.assertEqual(actual_state["transformation_md"], mocked_transformation_plan)
        self.assertTrue(mocked_transformation_plan.__str__.called)
        # Verify all parallel LLM calls were made
        self.assertTrue(self.mocked_metadata_llm.ainvoke.called)
        self.assertTrue(self.mocked_fields_llm.ainvoke.called)
        self.assertTrue(self.mocked_forward_llm.ainvoke.called)
        self.assertTrue(self.mocked_backward_llm.ainvoke.called)
        self.assertTrue(self.mocked_synch_llm.ainvoke.called)
        self.assertNotIn("iteration", actual_state)

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
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
        mock_write_text: Mock,
        mock_touch: Mock,
    ):
        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        plan_content = "This is the transformation plan."
        mocked_parser = Mock(spec=TransformationPlanParser)
        mocked_transformation_plan = Mock(spec=TransformationPlan)
        mocked_transformation_plan.__str__ = Mock(return_value=plan_content)

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: mocked_transformation_plan,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": mocked_transformation_plan,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        actual_state = asyncio.run(implement_transformation(state))

        # Verify that the metadata LLM was called with a prompt that includes the transformation plan
        call_args = self.mocked_metadata_llm.ainvoke.call_args
        prompt = call_args.kwargs.get("input") or call_args.args[0]
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", prompt)
        self.assertIn(plan_content, prompt)
        self.assertNotIn("iteration", actual_state)

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__derives_class_name_from_path_and_does_not_ask_llm(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
        mock_touch: Mock,
    ):
        """Anforderung 2: ``implement_transformation`` must not ask the LLM for a
        class name. The structured-output spec has no ``class_name`` field, and
        the name is derived from the ``transformation_class_path`` stem (set by
        ``prepare_workspace``) and forwarded to the template renderer.
        """
        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        asyncio.run(implement_transformation(state))

        # The LLMs are asked with specs that do NOT contain a class_name field.
        self.mocked_llm.with_structured_output.assert_any_call(
            TransformationClassMetadata
        )
        self.mocked_llm.with_structured_output.assert_any_call(
            TransformationFieldsAndConstructor
        )
        # The class name passed to the renderer is derived from the path stem.
        _, kwargs = mock_render_template.call_args
        self.assertEqual(kwargs.get("class_name"), "MyTransformation")


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
            "synch_body": 'System.out.println("synced");',
            "transform_source_to_target_body": "forward(source, target, decisions);",
            "transform_target_to_source_body": "backward(target, source, decisions);",
        }
        rendered = resolver.render_template(TransformationClassSpec(**fake_data))

        self.assertIn("implements AgentTransformationForEMF<", rendered)
        self.assertIn("SourceModel", rendered)
        self.assertIn("TargetModel", rendered)
        self.assertIn("TransformationDecisions", rendered)
        self.assertIn("public void forward(SourceModel source", rendered)


class TestPiecewiseGenerationPrompts(TestCase):
    """Tests for the helper functions that create prompts for piecewise generation."""

    def test_create_metadata_prompt__includes_all_required_sections(self):
        from mdeagent.implementation.implement_transformation import (
            create_metadata_prompt,
        )

        task_spec = "Transform families to persons."
        plan = "Step 1: Extract members."
        template = "public class {{ class_name }} {}"

        prompt = create_metadata_prompt(task_spec, plan, template)

        self.assertIn("--- BEGIN TASK SPECIFICATION ---", prompt)
        self.assertIn(task_spec, prompt)
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", prompt)
        self.assertIn(plan, prompt)
        self.assertIn("--- BEGIN TEMPLATE ---", prompt)
        self.assertIn(template, prompt)
        self.assertIn("Return the package name, source type, target type", prompt)

    def test_create_fields_and_constructor_prompt__includes_metadata_context(self):
        from mdeagent.implementation.generator import TransformationClassMetadata
        from mdeagent.implementation.implement_transformation import (
            create_fields_and_constructor_prompt,
        )

        metadata = TransformationClassMetadata(
            package_name="com.example",
            source_type="SourceModel",
            target_type="TargetModel",
            decision_type="Decisions",
            transformation_package="com.example.api",
        )
        task_spec = "Transform families to persons."
        plan = "Step 1: Extract members."
        template = "public class {{ class_name }} {}"

        prompt = create_fields_and_constructor_prompt(
            task_spec, plan, template, metadata
        )

        self.assertIn("--- BEGIN METADATA ---", prompt)
        self.assertIn("Package: com.example", prompt)
        self.assertIn("Source Type: SourceModel", prompt)
        self.assertIn("Target Type: TargetModel", prompt)
        self.assertIn("Return the field declarations and constructor", prompt)

    def test_create_method_body_prompts__include_fields_info(self):
        from mdeagent.implementation.generator import TransformationClassMetadata
        from mdeagent.implementation.implement_transformation import (
            create_backward_body_prompt,
            create_forward_body_prompt,
            create_synch_body_prompt,
        )

        metadata = TransformationClassMetadata(
            package_name="com.example",
            source_type="SourceModel",
            target_type="TargetModel",
            decision_type="Decisions",
            transformation_package="com.example.api",
        )
        fields_info = "Fields: [String label, int count]"
        task_spec = "Transform families to persons."
        plan = "Step 1: Extract members."
        template = "public class {{ class_name }} {}"

        forward_prompt = create_forward_body_prompt(
            task_spec, plan, template, metadata, fields_info
        )
        backward_prompt = create_backward_body_prompt(
            task_spec, plan, template, metadata, fields_info
        )
        synch_prompt = create_synch_body_prompt(
            task_spec, plan, template, metadata, fields_info
        )

        # All prompts should include the fields info
        for prompt, name in [
            (forward_prompt, "forward"),
            (backward_prompt, "backward"),
            (synch_prompt, "synch"),
        ]:
            with self.subTest(prompt_name=name):
                self.assertIn("--- BEGIN FIELDS ---", prompt)
                self.assertIn(fields_info, prompt)
                self.assertIn("Return only the Java code", prompt)

        # Each prompt should mention the correct direction
        self.assertIn("transforms from SourceModel to TargetModel", forward_prompt)
        self.assertIn("transforms from TargetModel to SourceModel", backward_prompt)
        self.assertIn(
            "incremental updates between SourceModel and TargetModel", synch_prompt
        )


class TestParallelExecution(TestCase):
    """Tests verifying that the piecewise generation executes calls in parallel."""

    def setUp(self):
        self.fake_metadata = {
            "package_name": "com.example.transformation",
            "source_type": "SourceModel",
            "target_type": "TargetModel",
            "decision_type": "TransformationDecisions",
            "transformation_package": "com.example",
        }

        self.fake_fields_constructor = {
            "fields": [{"type": "String", "name": "label"}],
            "constructor": None,
        }

        self.fake_forward_body = {"forward_body": "// forward"}
        self.fake_backward_body = {"backward_body": "// backward"}
        self.fake_synch_body = {"synch_body": "// synch"}

        self.mocked_llm = Mock(spec=BaseChatModel)

        # Create separate mocks for each structured output LLM
        self.mocked_metadata_llm = Mock(spec=BaseChatModel)
        self.mocked_fields_llm = Mock(spec=BaseChatModel)
        self.mocked_forward_llm = Mock(spec=BaseChatModel)
        self.mocked_backward_llm = Mock(spec=BaseChatModel)
        self.mocked_synch_llm = Mock(spec=BaseChatModel)

        def mock_with_structured_output(schema):
            from mdeagent.implementation.generator import (
                BackwardMethodBody,
                ForwardMethodBody,
                SynchMethodBody,
                TransformationClassMetadata,
                TransformationFieldsAndConstructor,
            )

            if schema == TransformationClassMetadata:
                return self.mocked_metadata_llm
            elif schema == TransformationFieldsAndConstructor:
                return self.mocked_fields_llm
            elif schema == ForwardMethodBody:
                return self.mocked_forward_llm
            elif schema == BackwardMethodBody:
                return self.mocked_backward_llm
            elif schema == SynchMethodBody:
                return self.mocked_synch_llm
            else:
                fallback_mock = Mock(spec=BaseChatModel)
                return fallback_mock

        self.mocked_llm.with_structured_output = Mock(
            side_effect=mock_with_structured_output
        )

        # Set up ainvoke mocks
        self.mocked_metadata_llm.ainvoke = AsyncMock(
            return_value=TransformationClassMetadata(**self.fake_metadata)
        )
        self.mocked_fields_llm.ainvoke = AsyncMock(
            return_value=TransformationFieldsAndConstructor(
                **self.fake_fields_constructor
            )
        )
        self.mocked_forward_llm.ainvoke = AsyncMock(
            return_value=ForwardMethodBody(**self.fake_forward_body)
        )
        self.mocked_backward_llm.ainvoke = AsyncMock(
            return_value=BackwardMethodBody(**self.fake_backward_body)
        )
        self.mocked_synch_llm.ainvoke = AsyncMock(
            return_value=SynchMethodBody(**self.fake_synch_body)
        )

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template",
        return_value="template",
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template",
        return_value="code",
    )
    def test_piecewise_generation__calls_parallel_llms_after_metadata(
        self, mock_render, mock_get_raw, mock_write, mock_touch
    ):
        """Verifies that the four body-generation LLMs are called AFTER metadata,
        and that they are all called (indicating parallel execution via asyncio.gather)."""
        from mdeagent.implementation.implement_transformation import (
            create_implement_transformation_node,
        )

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        state: ImplementationState = {
            "task_specification": "Test spec",
            "transformation_md": Mock(__str__=lambda self: "plan"),
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/Test.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        asyncio.run(implement_transformation(state))

        # Metadata should be called first
        self.mocked_metadata_llm.ainvoke.assert_called_once()
        metadata_call_order = self.mocked_metadata_llm.ainvoke.call_args_list[0]

        # All four parallel LLMs should be called
        self.assertEqual(self.mocked_fields_llm.ainvoke.call_count, 1)
        self.assertEqual(self.mocked_forward_llm.ainvoke.call_count, 1)
        self.assertEqual(self.mocked_backward_llm.ainvoke.call_count, 1)
        self.assertEqual(self.mocked_synch_llm.ainvoke.call_count, 1)

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template",
        return_value="template",
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template",
        return_value="code",
    )
    def test_piecewise_generation__passes_metadata_to_subsequent_prompts(
        self, mock_render, mock_get_raw, mock_write, mock_touch
    ):
        """Verifies that metadata values are passed to the field and body prompts."""
        from mdeagent.implementation.implement_transformation import (
            create_implement_transformation_node,
        )

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        state: ImplementationState = {
            "task_specification": "Test spec",
            "transformation_md": Mock(__str__=lambda self: "plan"),
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/Test.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        asyncio.run(implement_transformation(state))

        # Check that the fields prompt received metadata
        fields_call = self.mocked_fields_llm.ainvoke.call_args
        fields_prompt = fields_call.kwargs.get("input") or fields_call.args[0]
        self.assertIn("Package: com.example.transformation", fields_prompt)
        self.assertIn("Source Type: SourceModel", fields_prompt)

        # Check that the body prompts received metadata
        for llm_mock in [
            self.mocked_forward_llm,
            self.mocked_backward_llm,
            self.mocked_synch_llm,
        ]:
            call = llm_mock.ainvoke.call_args
            prompt = call.kwargs.get("input") or call.args[0]
            self.assertIn("Source Type: SourceModel", prompt)
            self.assertIn("Target Type: TargetModel", prompt)
