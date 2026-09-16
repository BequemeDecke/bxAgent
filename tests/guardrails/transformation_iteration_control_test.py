"""
This test checks if the transformation iteration control node correctly limits the number of iterations the agent performs when trying to implement a model transformation.
"""

import asyncio
from datetime import datetime, timedelta, UTC
from unittest import TestCase
from unittest.mock import Mock

from langchain.messages import HumanMessage, SystemMessage

from mdeagent.evaluation.types import EvaluationError, EvaluationResult, EvaluationRun
from mdeagent.guardrails.transformation_iteration_control import (
    IterationRoute,
    create_check_transformation_iteration_function,
)
from mdeagent.state import MDEAgentState


class TestTransformationIterationControl(TestCase):
    def setUp(self):
        self.check_transformation_iteration = (
            create_check_transformation_iteration_function()
        )

    def test_transformation_iteration_control__stop_on_max_iterations(self):
        max_iterations = 3
        state: MDEAgentState = {
            "iteration": 3,
            "latest_evaluation_runs": [],
        }
        result = asyncio.run(self.check_transformation_iteration(state, max_iterations))
        self.assertEqual(result, "max_iteration_reached")

    def test_transformation_iteration_control__run_results_have_execution_errors(self):
        max_iterations = 3
        state: MDEAgentState = {
            "iteration": 2,
            "latest_evaluation_runs": [
                EvaluationRun(
                    started_at=datetime.now(tz=UTC) - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    results=[
                        EvaluationResult(
                            content="Evaluation result content",
                        )
                    ],
                    errors=[
                        EvaluationError(
                            message="An error occurred during the evaluation.",
                            type="EvaluationError",
                            details={
                                "error_code": "AUDIT_ERROR",
                                "severity": "high",
                            },
                        )
                    ],
                )
            ],
        }
        result = asyncio.run(self.check_transformation_iteration(state, max_iterations))
        self.assertEqual(result, "error")

    def test_transformation_iteration_control__run_results_have_design_errors(self):
        max_iterations = 3
        state: MDEAgentState = {
            "iteration": 2,
            "latest_evaluation_runs": [
                EvaluationRun(
                    started_at=datetime.now(tz=UTC) - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    category="design",
                    results=[
                        EvaluationResult(
                            content="Evaluation result content",
                            metadata={"success": False},
                        )
                    ],
                    errors=[],
                )
            ],
        }
        result = asyncio.run(self.check_transformation_iteration(state, max_iterations))
        self.assertEqual(result, "design_failed")

    def test_transformation_iteration_control__run_results_no_errors(self):
        max_iterations = 3
        state: MDEAgentState = {
            "transformation_source_model_description": "A model that needs to be transformed.",
            "transformation_target_model_description": "The desired model after transformation.",
            "iteration": 2,
            "latest_evaluation_runs": [
                EvaluationRun(
                    started_at=datetime.now(tz=UTC) - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    category="design",
                    results=[
                        EvaluationResult(
                            content="Evaluation result content",
                            metadata={"include_in_report": True, "success": True},
                        ),
                        EvaluationResult(
                            content="Another evaluation result content",
                            metadata={"include_in_report": False, "success": True},
                        ),
                        EvaluationResult(
                            content="Evaluation result with error",
                            metadata={"include_in_report": True, "success": True},
                        ),
                    ],
                    errors=[],
                ),
                EvaluationRun(
                    started_at=datetime.now(tz=UTC) - timedelta(minutes=5),
                    execution_time_ms=200,
                    iteration=1,
                    category="execution",
                    results=[
                        EvaluationResult(
                            content="Another evaluation result content",
                            metadata={"include_in_report": False, "success": True},
                        ),
                        EvaluationResult(
                            content="Evaluation result with error",
                            metadata={"include_in_report": True, "success": True},
                        ),
                    ],
                    errors=[],
                ),
            ],
        }
        result = asyncio.run(self.check_transformation_iteration(state, max_iterations))
        self.assertEqual(result, "design_passed")
