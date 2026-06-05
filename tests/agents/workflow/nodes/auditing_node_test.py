"""
This test checks if the auditing node correctly executes the auditing core and returns the expected results.

It is more of an integration test that checks the interaction between the auditing node and the auditing core, rather than a unit test of the auditing node itself.
"""

import asyncio

from unittest import TestCase
from unittest.mock import Mock

from bxagent.tools.audit.types import Audit, AuditResult, AuditError
from bxagent.tools.audit import AuditExecutor
from bxagent.agents.workflow.nodes.auditing_node import (
    create_audit_agent_work_function,
    StateToAuditParameterMapper,
)
from bxagent.agents.workflow.state import WorkflowState


class TestAuditingNode(TestCase):
    def test_auditing_node__updates_state_with_latest_results(self):
        mocked_mapper = Mock(spec=StateToAuditParameterMapper)
        mocked_mapper.map.return_value = {"mapped_param": "some_value"}

        mocked_audit = Mock(spec=Audit)
        mocked_audit.run.return_value = (
            [AuditResult(content="Some audit result")],
            [AuditError(message="Some audit error")],
        )

        audit_executor = AuditExecutor(audits={"test_audit": mocked_audit})
        audit_agent_work = create_audit_agent_work_function(
            audit_executor=audit_executor,
            mapper=[mocked_mapper],
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
            mocked_mapper.map.called, "The mapper's map method should have been called."
        )
        self.assertTrue(
            mocked_audit.run.called, "The audit's run method should have been called."
        )
        self.assertEqual(
            mocked_audit.run.call_args.kwargs,
            {"mapped_param": "some_value"},
            "The audit should have been called with the mapped parameters.",
        )
