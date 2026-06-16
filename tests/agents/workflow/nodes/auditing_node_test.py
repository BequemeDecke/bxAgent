"""
This test checks if the auditing node correctly executes the auditing core and returns the expected results.

It is more of an integration test that checks the interaction between the auditing node and the auditing core, rather than a unit test of the auditing node itself.
"""

import asyncio

from unittest import TestCase
from unittest.mock import Mock
from typing import Dict, Any
from pydantic import BaseModel

from bxagent.tools.validation.types import Audit, ValidationResult, ValidationError, StateToAuditMapper
from bxagent.tools.validation import ValidationExecutor
from bxagent.agents.workflow.nodes.auditing_node import (
    create_audit_agent_work_function,
)
from bxagent.agents.workflow.state import WorkflowState


class MockedSchema(BaseModel):
    some_field: str


class TestAuditingNode__ExecutionModeAll(TestCase):
    def test_auditing_node__updates_state_with_latest_results(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_audit = Mock(spec=Audit)
        mocked_audit.run.return_value = (
            [ValidationResult(content="Some audit result")],
            [ValidationError(message="Some audit error")],
        )

        audit_id = "test_audit"

        audit_executor = ValidationExecutor(
            audits={audit_id: {"audit": mocked_audit, "audit_schema": MockedSchema}}
        )
        audit_agent_work = create_audit_agent_work_function(
            audit_executor=audit_executor,
            mapper={audit_id: mocked_mapper},
            execution_mode="all",
        )

        result = asyncio.run(audit_agent_work(WorkflowState()))

        self.assertEqual(
            len(result["latest_audit_runs"]),
            1,
            "There should be results for one audit run.",
        )

        run_result = result["latest_audit_runs"][0]
        self.assertEqual(
            len(run_result.results), 1, "There should be one audit result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some audit result",
            "The audit result content should match.",
        )
        self.assertEqual(len(run_result.errors), 1, "There should be one audit error.")
        self.assertEqual(
            run_result.errors[0].message,
            "Some audit error",
            "The audit error message should match.",
        )

        self.assertTrue(
            mocked_audit.run.called, "The audit's run method should have been called."
        )
        self.assertEqual(
            mocked_audit.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The audit should have been called with the mapped parameters.",
        )

    def test_auditing_node__audit_has_no_mapper(self):
        mocked_audit = Mock(spec=Audit)
        mocked_audit.run.return_value = (
            [ValidationResult(content="Some audit result")],
            [ValidationError(message="Some audit error")],
        )

        audit_id = "test_audit"

        audit_executor = ValidationExecutor(
            audits={audit_id: {"audit": mocked_audit, "audit_schema": MockedSchema}}
        )
        audit_agent_work = create_audit_agent_work_function(
            audit_executor=audit_executor,
            mapper={},  # No mapper provided
            execution_mode="all",
        )

        with self.assertRaises(
            KeyError,
            msg="A KeyError should be raised when no mapper is provided for the audit.",
        ):
            asyncio.run(audit_agent_work(WorkflowState()))


class TestAuditingNode__ExecutionModeSpecific(TestCase):
    def test_auditing_node__updates_state_with_latest_results_specific(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_audit = Mock(spec=Audit)
        mocked_audit.run.return_value = (
            [ValidationResult(content="Some audit result")],
            [ValidationError(message="Some audit error")],
        )

        audit_id = "test_audit"

        audit_executor = ValidationExecutor(
            audits={audit_id: {"audit": mocked_audit, "audit_schema": MockedSchema}}
        )
        audit_agent_work = create_audit_agent_work_function(
            audit_executor=audit_executor,
            mapper={audit_id: mocked_mapper},
            execution_mode="specific",
        )

        result = asyncio.run(audit_agent_work(WorkflowState()))

        self.assertIn(
            audit_id,
            result["latest_audit_runs"],
            "The latest audit runs should contain the specific audit id.",
        )

        run_result = result["latest_audit_runs"][audit_id]
        self.assertEqual(
            len(run_result.results), 1, "There should be one audit result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some audit result",
            "The audit result content should match.",
        )
        self.assertEqual(len(run_result.errors), 1, "There should be one audit error.")
        self.assertEqual(
            run_result.errors[0].message,
            "Some audit error",
            "The audit error message should match.",
        )

        self.assertTrue(
            mocked_audit.run.called, "The audit's run method should have been called."
        )
        self.assertEqual(
            mocked_audit.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The audit should have been called with the mapped parameters.",
        )

    def test_auditing_node__audit_has_no_mapper_specific(self):
        mocked_audit = Mock(spec=Audit)
        mocked_audit.run.return_value = (
            [ValidationResult(content="Some audit result")],
            [ValidationError(message="Some audit error")],
        )

        audit_id = "test_audit"

        audit_executor = ValidationExecutor(
            audits={audit_id: {"audit": mocked_audit, "audit_schema": MockedSchema}}
        )
        audit_agent_work = create_audit_agent_work_function(
            audit_executor=audit_executor,
            mapper={},  # No mapper provided
            execution_mode="specific",
        )

        self.assertEqual(
            asyncio.run(audit_agent_work(WorkflowState())),
            {"latest_audit_runs": {}},
            "When no mapper is provided for specific execution mode, the latest audit runs should be empty.",
        )
