"""
Test cases for the implement_transformation node.

This node is part of the coding agent built with langgraph and is responsible for implementing the transformation logic based on the provided specifications and requirements.

It should read the `TRANSFORMATION.md` file for necessary information and use that to generate the appropriate java code for the transformation.

This component uses a few llm calls which will make testing more difficult. Therefore the tests consists of two types:
1. Unit tests: These tests will mock the llm calls and test the logic of the node in isolation.
2. Agent Evaluation: This will be an end-to-end test where the node is tested as part of the entire agent.
"""

from unittest import TestCase
from unittest.mock import patch

from bxagent.tools.coding.implement_transformation import (
    create_implement_transformation_node,
    create_input_prompt,
)


class TestCreateInputPrompt(TestCase):
    def test_create_input_prompt__returns_correctly_formatted_prompt(self):
        task_specification = (
            "Implement the transformation from model Anton to model Berta."
        )
        transformation_plan = "This is the transformation plan."

        actual_prompt = create_input_prompt(task_specification, transformation_plan)

        self.assertIn("--- BEGIN TASK SPECIFICATION ---", actual_prompt)
        self.assertIn(task_specification, actual_prompt)
        self.assertIn("--- BEGIN TRANSFORMATION PLAN ---", actual_prompt)
        self.assertIn(transformation_plan, actual_prompt)


class TestImplementTransformation(TestCase):
    def test_implement_transformation__return_compiled_code(self):
        self.assertTrue(False)
