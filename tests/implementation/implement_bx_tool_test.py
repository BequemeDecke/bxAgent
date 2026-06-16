"""
Test for the implementation of the bx tool. The test should check if the tool correctly integrates the transformation logic into a bx tool using a provided template.

Two types of tests should be implemented:
1. Unit tests: These tests will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent, ensuring that the generated bx tool works correctly within the agent
"""

from unittest import TestCase
from unittest.mock import Mock, patch
from langchain.chat_models import BaseChatModel
from pathlib import Path

from bxagent.implementation.implement_bx_tool import (
    create_implement_bx_tool_node,
    create_input_prompt,
)
from bxagent.implementation.state import CodingAgentState
from bxagent.comprehension.bxtool import BxToolForEMF


class TestImplementBxTool(TestCase):
    def setUp(self):
        # --- Data ---
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

        # --- Mocks ---
        mocked_llm = Mock(spec=BaseChatModel)
        mocked_llm_structured_output = Mock(spec=BaseChatModel)
        mocked_llm.with_structured_output.return_value = mocked_llm_structured_output
        mocked_llm_structured_output.invoke.return_value = BxToolForEMF(
            **self.fake_data
        )

        # --- Node ---
        self.implement_bx_tool = create_implement_bx_tool_node(
            mocked_llm, Path("/tmp/workspace")
        )

    @patch("pathlib.Path.write_text")
    @patch(
        "bxagent.tools.transformation.bxtool.BxToolTemplateResolver.get_raw_template"
    )
    @patch("bxagent.tools.transformation.bxtool.BxToolTemplateResolver.render_template")
    def test_implement_bx_tool__write_bxtool_implementation(
        self,
        mock_render_template: Mock,
        mock_get_raw_template: Mock,
        mock_write_text: Mock,
    ):
        mock_write_text.return_value = None  # Mock the write_text method to do nothing
        mock_get_raw_template.return_value = "Raw template content"
        mock_render_template.return_value = "Rendered bx tool content"

        state = CodingAgentState(
            transformation_md=None,
            task_specification="Implement the bx tool",
            written_java_files=[],
            transformation_implementation="public class MyTransformation { ... }",
        )

        new_state = self.implement_bx_tool(state)

        # Check if the new state contains the path to the written Java file
        actual_written_files = new_state["written_java_files"]
        self.assertEqual(len(actual_written_files), 1)
        self.assertEqual(
            actual_written_files[0],
            Path("/tmp/workspace/TransformationImplementation.java"),
        )

        # Check if the template resolver methods were called with the correct parameters
        mock_get_raw_template.assert_called_once()
        mock_render_template.assert_called_once_with(BxToolForEMF(**self.fake_data))
