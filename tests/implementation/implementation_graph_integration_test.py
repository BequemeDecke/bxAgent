"""
Integration tests for the implement_transformation node.

These tests verify that the implement_transformation node works correctly
as part of the implementation graph, without mocking the full graph structure.
"""

import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from mdeagent.comprehension.plan import FileTransformationPlanParser, TransformationPlan
from mdeagent.implementation.generator import TransformationClassSpec
from mdeagent.implementation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.state import ImplementationState


class TestImplementTransformationIntegration(TestCase):
    """Integration tests for the implement_transformation node."""

    @patch("mdeagent.models.build_base_model")
    def test_implement_transformation_node__creates_transformation_file(
        self, mock_build_base_model
    ):
        """
        Test that the implement_transformation node correctly creates a transformation file.
        
        This test:
        1. Creates the implement_transformation node with a real LLM
        2. Invokes the node with a state containing a transformation plan
        3. Checks that a TransformationClass.java file is created
        4. Verifies the state is correctly updated
        """
        # Create a mock LLM that returns a structured transformation class spec
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = TransformationClassSpec(
            package_name="com.example.transformation",
            class_name="SourceToTargetTransformation",
            source_type="SourceModel",
            target_type="TargetModel",
            decision_type="TransformationDecisions",
            fields=[],
            forward_body='System.out.println("Forward transformation");',
        )
        mock_build_base_model.return_value = mock_llm

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a transformation plan file
            tp_file = workspace / "TRANSFORMATION.md"
            tp_file.write_text("""\
---
source_model_package: com.example.source
target_model_package: com.example.target
iteration: 1
---

# Transformation Plan

## 1. Model Implementations

--- BEGIN SOURCE MODEL ---
public class SourceModel { }
--- END SOURCE MODEL ---

--- BEGIN TARGET MODEL ---
public class TargetModel { }
--- END TARGET MODEL ---

## 2. Transformation Direction

Bidirectional transformation
--- BEGIN TRANSFORMATION DIRECTION ---
forward and backward
--- END TRANSFORMATION DIRECTION ---

## 3. Identified Difficulties

None
--- BEGIN DIFFICULTIES ---
None identified
--- END DIFFICULTIES ---

## 4. Implementation Steps

1. Create the transformation class
2. Implement the forward method
--- BEGIN IMPLEMENTATION STEPS ---
1. Create the transformation class
2. Implement the forward method
--- END IMPLEMENTATION STEPS ---
""")

            # Create the node
            node = create_implement_transformation_node(
                llm=mock_build_base_model(),
                workspace=workspace,
                optional_plan_factory=lambda: TransformationPlan.parse(
                    FileTransformationPlanParser(tp_file)
                ),
            )

            # Create the initial state
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(tp_file)
            )
            initial_state = ImplementationState(
                transformation_md=transformation_plan,
                task_specification="Implement the transformation from SourceModel to TargetModel.",
                written_java_files=[],
                bxtool_path=workspace / "BxTool.java",
                transformation_implementation="",
                latest_evaluation_results={},
                implementation_iteration=1,
            )

            # Invoke the node
            result = node(initial_state)

            # Assertions
            self.assertIn("written_java_files", result)
            self.assertGreater(len(result["written_java_files"]), 0)
            
            # Check that the transformation file exists
            transformation_file = result["written_java_files"][0]
            self.assertTrue(transformation_file.exists())
            self.assertEqual(transformation_file.suffix, ".java")
            
            # Check the file contains the expected class name
            file_content = transformation_file.read_text()
            self.assertIn("SourceToTargetTransformation", file_content)
            self.assertIn("implements AgentTransformationForEMF", file_content)

    @patch("mdeagent.models.build_base_model")
    def test_implement_transformation_node__appends_to_existing_files(
        self, mock_build_base_model
    ):
        """
        Test that the implement_transformation node correctly appends to existing files.
        """
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = TransformationClassSpec(
            package_name="com.example.transformation",
            class_name="NewTransformation",
            source_type="SourceModel",
            target_type="TargetModel",
            decision_type="TransformationDecisions",
            fields=[],
        )
        mock_build_base_model.return_value = mock_llm

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create an existing file
            existing_file = workspace / "ExistingTransformation.java"
            existing_file.write_text("public class ExistingTransformation { }")
            
            # Create a transformation plan file
            tp_file = workspace / "TRANSFORMATION.md"
            tp_file.write_text("""\
---
source_model_package: com.example.source
target_model_package: com.example.target
iteration: 1
---

--- BEGIN SOURCE MODEL ---
public class SourceModel { }
--- END SOURCE MODEL ---

--- BEGIN TARGET MODEL ---
public class TargetModel { }
--- END TARGET MODEL ---

--- BEGIN TRANSFORMATION DIRECTION ---
forward and backward
--- END TRANSFORMATION DIRECTION ---

--- BEGIN DIFFICULTIES ---
None
--- END DIFFICULTIES ---

--- BEGIN IMPLEMENTATION STEPS ---
1. Create the transformation class
--- END IMPLEMENTATION STEPS ---
""")

            # Create the node
            node = create_implement_transformation_node(
                llm=mock_build_base_model(),
                workspace=workspace,
                optional_plan_factory=lambda: TransformationPlan.parse(
                    FileTransformationPlanParser(tp_file)
                ),
            )

            # Create the initial state with existing file
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(tp_file)
            )
            initial_state = ImplementationState(
                transformation_md=transformation_plan,
                task_specification="Implement a new transformation.",
                written_java_files=[existing_file],
                bxtool_path=workspace / "BxTool.java",
                transformation_implementation="",
                latest_evaluation_results={},
                implementation_iteration=1,
            )

            # Invoke the node
            result = node(initial_state)

            # Assertions
            self.assertIn("written_java_files", result)
            self.assertEqual(len(result["written_java_files"]), 2)
            self.assertIn(existing_file, result["written_java_files"])
            self.assertTrue(result["written_java_files"][1].exists())

    @patch("mdeagent.models.build_base_model")
    def test_implement_transformation_node__stores_transformation_implementation_in_state(
        self, mock_build_base_model
    ):
        """
        Test that the implement_transformation node stores the transformation implementation in the state.
        """
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = TransformationClassSpec(
            package_name="com.example.transformation",
            class_name="TestTransformation",
            source_type="SourceModel",
            target_type="TargetModel",
            decision_type="Decisions",
            fields=[],
            forward_body='System.out.println("Forward");',
        )
        mock_build_base_model.return_value = mock_llm

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a transformation plan file
            tp_file = workspace / "TRANSFORMATION.md"
            tp_file.write_text("""\
---
source_model_package: com.example.source
target_model_package: com.example.target
iteration: 1
---

--- BEGIN SOURCE MODEL ---
public class SourceModel { }
--- END SOURCE MODEL ---

--- BEGIN TARGET MODEL ---
public class TargetModel { }
--- END TARGET MODEL ---

--- BEGIN TRANSFORMATION DIRECTION ---
forward
--- END TRANSFORMATION DIRECTION ---

--- BEGIN DIFFICULTIES ---
None
--- END DIFFICULTIES ---

--- BEGIN IMPLEMENTATION STEPS ---
Implement forward
--- END IMPLEMENTATION STEPS ---
""")

            # Create the node
            node = create_implement_transformation_node(
                llm=mock_build_base_model(),
                workspace=workspace,
                optional_plan_factory=lambda: TransformationPlan.parse(
                    FileTransformationPlanParser(tp_file)
                ),
            )

            # Create the initial state
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(tp_file)
            )
            initial_state = ImplementationState(
                transformation_md=transformation_plan,
                task_specification="Implement the transformation.",
                written_java_files=[],
                bxtool_path=workspace / "BxTool.java",
                transformation_implementation="",
                latest_evaluation_results={},
                implementation_iteration=1,
            )

            # Invoke the node
            result = node(initial_state)

            # Assertions
            self.assertIn("transformation_implementation", result)
            self.assertIsNotNone(result["transformation_implementation"])
            self.assertIn("TestTransformation", result["transformation_implementation"])

    @patch("mdeagent.models.build_base_model")
    def test_implement_transformation_node__includes_plan_in_prompt(
        self, mock_build_base_model
    ):
        """
        Test that the implement_transformation node includes the transformation plan in the LLM prompt.
        """
        mock_llm = Mock()
        mock_structured_llm = Mock()
        mock_llm.with_structured_output.return_value = mock_structured_llm
        mock_structured_llm.invoke.return_value = TransformationClassSpec(
            package_name="com.example",
            class_name="TestTransformation",
            source_type="Source",
            target_type="Target",
            decision_type="Decisions",
            fields=[],
        )
        mock_build_base_model.return_value = mock_llm

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create a transformation plan file
            tp_file = workspace / "TRANSFORMATION.md"
            plan_content = """\
---
source_model_package: com.example.source
target_model_package: com.example.target
iteration: 1
---

--- BEGIN SOURCE MODEL ---
public class SourceModel { }
--- END SOURCE MODEL ---

--- BEGIN TARGET MODEL ---
public class TargetModel { }
--- END TARGET MODEL ---

--- BEGIN TRANSFORMATION DIRECTION ---
forward
--- END TRANSFORMATION DIRECTION ---

--- BEGIN DIFFICULTIES ---
None
--- END DIFFICULTIES ---

--- BEGIN IMPLEMENTATION STEPS ---
Implement the transformation
--- END IMPLEMENTATION STEPS ---
"""
            tp_file.write_text(plan_content)

            # Create the node
            node = create_implement_transformation_node(
                llm=mock_build_base_model(),
                workspace=workspace,
                optional_plan_factory=lambda: TransformationPlan.parse(
                    FileTransformationPlanParser(tp_file)
                ),
            )

            # Create the initial state
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(tp_file)
            )
            initial_state = ImplementationState(
                transformation_md=transformation_plan,
                task_specification="Implement the transformation.",
                written_java_files=[],
                bxtool_path=workspace / "BxTool.java",
                transformation_implementation="",
                latest_evaluation_results={},
                implementation_iteration=1,
            )

            # Invoke the node
            node(initial_state)

            # Verify that the LLM was called with a prompt that includes the transformation plan
            call_args = mock_structured_llm.invoke.call_args
            prompt = call_args.kwargs.get("input") or call_args.args[0]
            self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", prompt)
            self.assertIn("--- END TRANSFORMATION PLAN ---", prompt)
            self.assertIn("SOURCE MODEL", prompt)
            self.assertIn("TARGET MODEL", prompt)