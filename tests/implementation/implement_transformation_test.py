"""
Test cases for the implement_transformation node.

This node is part of the coding agent built with langgraph and is responsible for implementing the transformation logic based on the provided specifications and requirements.

It should read the `TRANSFORMATION.md` file for necessary information and use that to generate the appropriate java code for the transformation.

This component uses a few llm calls which will make testing more difficult. Therefore the tests consists of two types:
1. Unit tests: These will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from langchain.chat_models import BaseChatModel

from mdeagent.comprehension.plan import TransformationPlan, TransformationPlanParser
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun
from mdeagent.evaluation.utils import _format_evaluation_results, filter_execution_results
from mdeagent.implementation.transformation.generator import (
    BackwardMethodBody,
    ForwardMethodBody,
    SynchMethodBody,
    TransformationClassMetadata,
    TransformationClassSpec,
    TransformationClassTemplateResolver,
    TransformationFieldsAndConstructor,
)
from mdeagent.implementation.transformation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.transformation.prompts import create_backward_body_prompt, create_forward_body_prompt, create_input_prompt


class TestCreateInputPrompt(TestCase):
    def test_create_input_prompt__returns_correctly_formatted_prompt(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "This is the transformation plan."
        template = "public class {{ class_name }} { }"
        evaluation_results = "Compilation successful."

        actual_prompt = create_input_prompt(
            task_specification, transformation_plan, template, evaluation_results
        )

        self.assertIn("--- BEGIN TASK SPECIFICATION ---", actual_prompt)
        self.assertIn(task_specification, actual_prompt)
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", actual_prompt)
        self.assertIn(transformation_plan, actual_prompt)
        self.assertIn("--- BEGIN TEMPLATE ---", actual_prompt)
        self.assertIn(template, actual_prompt)
        self.assertIn("--- BEGIN EVALUATION RESULTS ---", actual_prompt)
        self.assertIn(evaluation_results, actual_prompt)

    def test_create_input_prompt__includes_difficulties_from_plan(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "--- BEGIN DIFFICULTIES ---\nSome difficulties here\n--- END DIFFICULTIES ---"
        template = "public class {{ class_name }} { }"
        evaluation_results = "No evaluation results available."

        actual_prompt = create_input_prompt(
            task_specification, transformation_plan, template, evaluation_results
        )

        self.assertIn("--- BEGIN DIFFICULTIES ---", transformation_plan)
        self.assertIn("Some difficulties here", actual_prompt)

    def test_create_input_prompt__includes_custom_evaluation_results(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "This is the transformation plan."
        template = "public class {{ class_name }} { }"
        evaluation_results_text = "1. [FAILURE] Compilation error in File.java:42"

        actual_prompt = create_input_prompt(
            task_specification,
            transformation_plan,
            template,
            evaluation_results_text=evaluation_results_text,
        )

        self.assertIn("--- BEGIN EVALUATION RESULTS ---", actual_prompt)
        self.assertIn(evaluation_results_text, actual_prompt)
        self.assertIn("--- END EVALUATION RESULTS ---", actual_prompt)


class TestFormatEvaluationResults(TestCase):
    def test_format_evaluation_results__empty_list(self):
        actual = _format_evaluation_results([])
        self.assertEqual(actual, "No evaluation results available.")

    def test_format_evaluation_results__single_success_result(self):
        results = [
            EvaluationResult(
                content="Maven project compiled successfully.",
                metadata={"success": True},
            )
        ]
        actual = _format_evaluation_results(results)
        self.assertIn("[SUCCESS]", actual)
        self.assertIn("Maven project compiled successfully.", actual)

    def test_format_evaluation_results__single_failure_result(self):
        results = [
            EvaluationResult(
                content="File does not exist: /tmp/Test.java",
                metadata={"success": False, "file": "/tmp/Test.java"},
            )
        ]
        actual = _format_evaluation_results(results)
        self.assertIn("[FAILURE]", actual)
        self.assertIn("File does not exist: /tmp/Test.java", actual)
        self.assertIn("File: /tmp/Test.java", actual)

    def test_format_evaluation_results__result_with_line_and_column(self):
        results = [
            EvaluationResult(
                content="Cannot find symbol",
                metadata={
                    "success": False,
                    "file": "/tmp/Test.java",
                    "line": 42,
                    "column": 15,
                },
            )
        ]
        actual = _format_evaluation_results(results)
        self.assertIn("Line: 42", actual)
        self.assertIn("Column: 15", actual)

    def test_format_evaluation_results__multiple_results(self):
        results = [
            EvaluationResult(
                content="Compilation error 1",
                metadata={"success": False, "line": 10},
            ),
            EvaluationResult(
                content="Compilation error 2",
                metadata={"success": False, "line": 20},
            ),
        ]
        actual = _format_evaluation_results(results)
        self.assertIn("1. [FAILURE]", actual)
        self.assertIn("2. [FAILURE]", actual)
        self.assertIn("Line: 10", actual)
        self.assertIn("Line: 20", actual)


class TestFilterExecutionResults(TestCase):
    def test_filter_execution_results__empty_dict(self):
        actual = filter_execution_results({})
        self.assertEqual(actual, [])

    def test_filter_execution_results__filters_by_execution_category(self):
        """Should only include results from runs with category 'execution'.

        Uses OR logic: includes results that are either errors (success=False)
        OR report candidates (include_in_report=True).
        """
        now = datetime.now()
        execution_run = EvaluationRun(
            started_at=now,
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[
                EvaluationResult(
                    content="Compilation failed",
                    metadata={"success": False, "include_in_report": True},
                ),
                EvaluationResult(
                    content="Success result",
                    metadata={"success": True, "include_in_report": True},
                ),
                EvaluationResult(
                    content="Error not for report",
                    metadata={"success": False, "include_in_report": False},
                ),
            ],
            errors=[],
        )
        design_run = EvaluationRun(
            started_at=now,
            execution_time_ms=50,
            iteration=1,
            category="design",
            results=[
                EvaluationResult(
                    content="Design issue",
                    metadata={"success": False, "include_in_report": True},
                ),
            ],
            errors=[],
        )

        latest_runs = {
            "java_compilation": execution_run,
            "workspace_structure": design_run,
        }

        actual = filter_execution_results(latest_runs)

        # Should only include results from execution run (3 results with OR logic):
        # 1. Compilation failed (error + report candidate)
        # 2. Success result (report candidate)
        # 3. Error not for report (error)
        self.assertEqual(len(actual), 3)
        result_contents = {r.content for r in actual}
        self.assertIn("Compilation failed", result_contents)
        self.assertIn("Success result", result_contents)
        self.assertIn("Error not for report", result_contents)
        # Design run results should NOT be included
        design_contents = {r.content for r in actual if "Design" in r.content}
        self.assertEqual(len(design_contents), 0)

    def test_filter_execution_results__includes_error_results(self):
        """IsErrorFilter should filter results with success=False."""
        now = datetime.now()
        execution_run = EvaluationRun(
            started_at=now,
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[
                EvaluationResult(
                    content="Success result",
                    metadata={"success": True, "include_in_report": False},
                ),
                EvaluationResult(
                    content="Error result",
                    metadata={"success": False, "include_in_report": True},
                ),
            ],
            errors=[],
        )

        latest_runs = {"java_compilation": execution_run}
        actual = filter_execution_results(latest_runs)

        # IsErrorFilter picks up the error result
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].content, "Error result")

    def test_filter_execution_results__includes_report_candidate_results(self):
        """IsReportCandidateFilter should filter results with include_in_report=True."""
        now = datetime.now()
        execution_run = EvaluationRun(
            started_at=now,
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[
                EvaluationResult(
                    content="Not in report",
                    metadata={"success": True, "include_in_report": False},
                ),
                EvaluationResult(
                    content="In report",
                    metadata={"success": False, "include_in_report": True},
                ),
            ],
            errors=[],
        )

        latest_runs = {"file_existence": execution_run}
        actual = filter_execution_results(latest_runs)

        # Both filters may pick up the result (it's both error and report candidate)
        # But deduplication ensures it appears only once
        self.assertGreaterEqual(len(actual), 1)
        self.assertIn("In report", [r.content for r in actual])

    def test_filter_execution_results__deduplicates_results(self):
        """Results that match both filters should appear only once."""
        now = datetime.now()
        execution_run = EvaluationRun(
            started_at=now,
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[
                EvaluationResult(
                    content="Both error and report",
                    metadata={"success": False, "include_in_report": True},
                ),
            ],
            errors=[],
        )

        latest_runs = {"java_compilation": execution_run}
        actual = filter_execution_results(latest_runs)

        # The same result matches both filters but should appear only once
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].content, "Both error and report")


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

        # Track call order for assertions
        self.llm_call_order = []

        # Helper to create a mock response with content attribute
        def make_response(content_str):
            mock_response = Mock()
            mock_response.content = content_str
            return mock_response

        # Create JSON responses for each expected call
        metadata_json = '{"package_name": "com.example.transformation", "source_type": "SourceModel", "target_type": "TargetModel", "decision_type": "TransformationDecisions", "transformation_package": "com.example"}'
        fields_json = '{"fields": [{"type": "String", "name": "label"}], "constructor": {"parameters": "String label", "assignments": [{"target": "label", "value": "label"}]}}'
        forward_json = '{"forward_body": "System.out.println(source);"}'
        backward_json = '{"backward_body": "System.out.println(target);"}'
        synch_json = '{"synch_body": "System.out.println(\\"synced\\");"}'

        # Configure ainvoke to return appropriate responses based on prompt content
        async def mock_ainvoke(prompt):
            self.llm_call_order.append(prompt)
            prompt_lower = str(prompt).lower()
            
            if "metadata" in prompt_lower or ("package" in prompt_lower and "source type" in prompt_lower):
                return make_response(metadata_json)
            elif "fields" in prompt_lower and "constructor" in prompt_lower:
                return make_response(fields_json)
            elif "forward" in prompt_lower and "body" in prompt_lower:
                return make_response(forward_json)
            elif "backward" in prompt_lower and "body" in prompt_lower:
                return make_response(backward_json)
            elif "synch" in prompt_lower or "synchronization" in prompt_lower:
                return make_response(synch_json)
            else:
                # Default to metadata response
                return make_response(metadata_json)

        self.mocked_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

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

        # Verify LLM was called 5 times (metadata + 4 parallel calls)
        self.assertEqual(self.mocked_llm.ainvoke.call_count, 5)

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
        # Verify all 5 parallel LLM calls were made
        self.assertEqual(self.mocked_llm.ainvoke.call_count, 5)
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

        # Verify that the LLM was called with a prompt that includes the transformation plan
        self.assertGreaterEqual(len(self.llm_call_order), 1)
        metadata_prompt = self.llm_call_order[0]  # First call is metadata
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", str(metadata_prompt))
        self.assertIn(plan_content, str(metadata_prompt))
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
            "latest_evaluation_runs": {},
            "iteration": 1,
        }

        asyncio.run(implement_transformation(state))

        # The class name passed to the renderer is derived from the path stem.
        _, kwargs = mock_render_template.call_args
        self.assertEqual(kwargs.get("class_name"), "MyTransformation")

    @patch("pathlib.Path.touch")
    @patch("pathlib.Path.write_text")
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.get_raw_template"
    )
    @patch(
        "mdeagent.implementation.generator.TransformationClassTemplateResolver.render_template"
    )
    def test_implement_transformation__includes_evaluation_results_in_prompt(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
        mock_touch: Mock,
    ):
        """Test that evaluation results from JavaCompilation and FileExistence are
        included in the prompt sent to the LLM."""
        from datetime import datetime

        mock_touch.return_value = None
        mock_write_text.return_value = None
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered transformation content"

        implement_transformation = create_implement_transformation_node(
            llm=self.mocked_llm,
            optional_plan_factory=lambda: None,
        )

        now = datetime.now()
        execution_run = EvaluationRun(
            started_at=now,
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[
                EvaluationResult(
                    content="File does not exist: /tmp/Test.java",
                    metadata={
                        "success": False,
                        "file": "/tmp/Test.java",
                        "include_in_report": True,
                    },
                ),
            ],
            errors=[],
        )

        state: ImplementationState = {
            "task_specification": "Implement the transformation from model Anton to model Berta.",
            "transformation_md": None,
            "written_java_files": [],
            "bxtool_path": Path("/tmp/workspace"),
            "transformation_class_path": Path("/tmp/workspace/MyTransformation.java"),
            "transformation_implementation": "",
            "latest_evaluation_runs": {"file_existence": execution_run},
            "iteration": 1,
        }

        asyncio.run(implement_transformation(state))

        # Verify that the LLM was called with a prompt that includes evaluation results
        # (The metadata call happens first and includes the evaluation results)
        self.assertGreaterEqual(len(self.llm_call_order), 1)
        metadata_prompt = self.llm_call_order[0]  # First call is metadata
        prompt_str = str(metadata_prompt)
        self.assertIn("--- BEGIN EVALUATION RESULTS ---", prompt_str)
        self.assertIn("[FAILURE]", prompt_str)
        self.assertIn("File does not exist: /tmp/Test.java", prompt_str)
        self.assertIn("File: /tmp/Test.java", prompt_str)


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
        from mdeagent.implementation.transformation.prompts import (
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
        from mdeagent.implementation.transformation.generator import TransformationClassMetadata
        from mdeagent.implementation.transformation.prompts import (
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
        from mdeagent.implementation.transformation.generator import TransformationClassMetadata
        from mdeagent.implementation.transformation.prompts import (
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

        self.mocked_llm = Mock(spec=BaseChatModel)
        self.llm_call_order = []
        self.call_timestamps = []  # Track when each call happens

        # Helper to create a mock response with content attribute
        def make_response(content_str):
            mock_response = Mock()
            mock_response.content = content_str
            return mock_response

        # JSON responses for each expected call
        metadata_json = '{"package_name": "com.example.transformation", "source_type": "SourceModel", "target_type": "TargetModel", "decision_type": "TransformationDecisions", "transformation_package": "com.example"}'
        fields_json = '{"fields": [{"type": "String", "name": "label"}], "constructor": null}'
        forward_json = '{"forward_body": "// forward"}'
        backward_json = '{"backward_body": "// backward"}'
        synch_json = '{"synch_body": "// synch"}'

        # Configure ainvoke to track calls and return appropriate responses
        import time
        async def mock_ainvoke(prompt):
            self.llm_call_order.append(prompt)
            self.call_timestamps.append(time.time())
            prompt_lower = str(prompt).lower()
            
            if "metadata" in prompt_lower or ("package" in prompt_lower and "source type" in prompt_lower):
                return make_response(metadata_json)
            elif "fields" in prompt_lower and "constructor" in prompt_lower:
                return make_response(fields_json)
            elif "forward" in prompt_lower and "body" in prompt_lower:
                return make_response(forward_json)
            elif "backward" in prompt_lower and "body" in prompt_lower:
                return make_response(backward_json)
            elif "synch" in prompt_lower or "synchronization" in prompt_lower:
                return make_response(synch_json)
            else:
                return make_response(metadata_json)

        self.mocked_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

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
        from mdeagent.implementation.transformation.implement_transformation import (
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

        # Should have exactly 5 calls (1 metadata + 4 parallel)
        self.assertEqual(self.mocked_llm.ainvoke.call_count, 5)
        
        # First call should be metadata (happens before parallel calls)
        self.assertGreaterEqual(len(self.llm_call_order), 5)
        first_prompt = str(self.llm_call_order[0]).lower()
        self.assertIn("metadata", first_prompt)

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
        from mdeagent.implementation.transformation.implement_transformation import (
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

        # Check that the fields prompt received metadata (call index 1, after metadata)
        self.assertGreaterEqual(len(self.llm_call_order), 5)
        fields_prompt = str(self.llm_call_order[1])
        self.assertIn("Package: com.example.transformation", fields_prompt)
        self.assertIn("Source Type: SourceModel", fields_prompt)

        # Check that the body prompts received metadata (calls 2-4)
        for i in range(2, 5):
            body_prompt = str(self.llm_call_order[i])
            self.assertIn("Source Type: SourceModel", body_prompt)
            self.assertIn("Target Type: TargetModel", body_prompt)
