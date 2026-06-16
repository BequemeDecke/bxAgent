"""
This test checks if the validationing node correctly executes the validationing core and returns the expected results.

It is more of an integration test that checks the interaction between the validationing node and the validationing core, rather than a unit test of the validationing node itself.
"""

import asyncio

from unittest import TestCase
from unittest.mock import Mock
from typing import Dict, Any
from pydantic import BaseModel

from bxagent.tools.validation.types import Validation, ValidationResult, ValidationError, StateToValidationMapper
from bxagent.tools.validation import ValidationExecutor
from bxagent.agents.workflow.nodes.validationing_node import (
    create_validation_agent_work_function,
)
from bxagent.agents.workflow.state import WorkflowState


class MockedSchema(BaseModel):
    some_field: str


class TestValidationingNode__ExecutionModeAll(TestCase):
    def test_validationing_node__updates_state_with_latest_results(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_validation = Mock(spec=Validation)
        mocked_validation.run.return_value = (
            [ValidationResult(content="Some validation result")],
            [ValidationError(message="Some validation error")],
        )

        validation_id = "test_validation"

        validation_executor = ValidationExecutor(
            validations={validation_id: {"validation": mocked_validation, "validation_schema": MockedSchema}}
        )
        validation_agent_work = create_validation_agent_work_function(
            validation_executor=validation_executor,
            mapper={validation_id: mocked_mapper},
            execution_mode="all",
        )

        result = asyncio.run(validation_agent_work(WorkflowState()))

        self.assertEqual(
            len(result["latest_validation_runs"]),
            1,
            "There should be results for one validation run.",
        )

        run_result = result["latest_validation_runs"][0]
        self.assertEqual(
            len(run_result.results), 1, "There should be one validation result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some validation result",
            "The validation result content should match.",
        )
        self.assertEqual(len(run_result.errors), 1, "There should be one validation error.")
        self.assertEqual(
            run_result.errors[0].message,
            "Some validation error",
            "The validation error message should match.",
        )

        self.assertTrue(
            mocked_validation.run.called, "The validation's run method should have been called."
        )
        self.assertEqual(
            mocked_validation.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The validation should have been called with the mapped parameters.",
        )

    def test_validationing_node__validation_has_no_mapper(self):
        mocked_validation = Mock(spec=Validation)
        mocked_validation.run.return_value = (
            [ValidationResult(content="Some validation result")],
            [ValidationError(message="Some validation error")],
        )

        validation_id = "test_validation"

        validation_executor = ValidationExecutor(
            validations={validation_id: {"validation": mocked_validation, "validation_schema": MockedSchema}}
        )
        validation_agent_work = create_validation_agent_work_function(
            validation_executor=validation_executor,
            mapper={},  # No mapper provided
            execution_mode="all",
        )

        with self.assertRaises(
            KeyError,
            msg="A KeyError should be raised when no mapper is provided for the validation.",
        ):
            asyncio.run(validation_agent_work(WorkflowState()))


class TestValidationingNode__ExecutionModeSpecific(TestCase):
    def test_validationing_node__updates_state_with_latest_results_specific(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_validation = Mock(spec=Validation)
        mocked_validation.run.return_value = (
            [ValidationResult(content="Some validation result")],
            [ValidationError(message="Some validation error")],
        )

        validation_id = "test_validation"

        validation_executor = ValidationExecutor(
            validations={validation_id: {"validation": mocked_validation, "validation_schema": MockedSchema}}
        )
        validation_agent_work = create_validation_agent_work_function(
            validation_executor=validation_executor,
            mapper={validation_id: mocked_mapper},
            execution_mode="specific",
        )

        result = asyncio.run(validation_agent_work(WorkflowState()))

        self.assertIn(
            validation_id,
            result["latest_validation_runs"],
            "The latest validation runs should contain the specific validation id.",
        )

        run_result = result["latest_validation_runs"][validation_id]
        self.assertEqual(
            len(run_result.results), 1, "There should be one validation result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some validation result",
            "The validation result content should match.",
        )
        self.assertEqual(len(run_result.errors), 1, "There should be one validation error.")
        self.assertEqual(
            run_result.errors[0].message,
            "Some validation error",
            "The validation error message should match.",
        )

        self.assertTrue(
            mocked_validation.run.called, "The validation's run method should have been called."
        )
        self.assertEqual(
            mocked_validation.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The validation should have been called with the mapped parameters.",
        )

    def test_validationing_node__validation_has_no_mapper_specific(self):
        mocked_validation = Mock(spec=Validation)
        mocked_validation.run.return_value = (
            [ValidationResult(content="Some validation result")],
            [ValidationError(message="Some validation error")],
        )

        validation_id = "test_validation"

        validation_executor = ValidationExecutor(
            validations={validation_id: {"validation": mocked_validation, "validation_schema": MockedSchema}}
        )
        validation_agent_work = create_validation_agent_work_function(
            validation_executor=validation_executor,
            mapper={},  # No mapper provided
            execution_mode="specific",
        )

        self.assertEqual(
            asyncio.run(validation_agent_work(WorkflowState())),
            {"latest_validation_runs": {}},
            "When no mapper is provided for specific execution mode, the latest validation runs should be empty.",
        )
