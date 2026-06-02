"""
This test checks if the transformation iteration control node correctly limits the number of iterations the agent performs when trying to implement a model transformation.
"""

from datetime import datetime, timedelta
from unittest import TestCase

from bxagent.agents.workflow.state import WorkflowState
from bxagent.agents.workflow.transformation_iteration_control import (
    check_transformation_iteration,
)
from bxagent.tools.audit.types import AuditRun, AuditResult, AuditError


class TestTransformationIterationControl(TestCase):
    def test_transformation_iteration_control__stop_on_max_iterations(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 3,
            "latest_audit_runs": [],
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "stop")

    def test_transformation_iteration_control__continue_before_max_iterations(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_audit_runs": [],
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "continue")

    def test_transformation_iteration_control__run_results_have_errors(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_audit_runs": [
                AuditRun(
                    started_at=datetime.now() - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    results=[
                        AuditResult(
                            content="Audit result content",
                        )
                    ],
                    errors=[
                        AuditError(
                            message="An error occurred during the audit.",
                            details={
                                "error_code": "AUDIT_ERROR",
                                "severity": "high",
                            },
                        )
                    ],
                )
            ],
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "continue")

    def test_transformation_iteration_control__run_results_no_errors(self):
        max_iterations = 3
        state: WorkflowState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_audit_runs": [
                AuditRun(
                    started_at=datetime.now() - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    results=[
                        AuditResult(
                            content="Audit result content",
                        )
                    ],
                    errors=[],
                )
            ],
        }
        result = check_transformation_iteration(state, max_iterations)
        self.assertEqual(result, "stop")
