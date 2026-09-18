"""
This test checks if the transformation iteration control node correctly limits the number of iterations the agent performs when trying to implement a model transformation.
"""

import asyncio
from pathlib import Path
import tempfile
from datetime import UTC, datetime, timedelta
from unittest import TestCase

from mdeagent.comprehension.plan import TransformationPlan, FileTransformationPlanParser
from mdeagent.evaluation.types import EvaluationError, EvaluationResult, EvaluationRun
from mdeagent.guardrails.transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from mdeagent.state import MDEAgentState


class TestTransformationIterationControl(TestCase):
    def setUp(self):
        self.check_transformation_iteration = (
            create_check_transformation_iteration_function()
        )

    def test_transformation_iteration_control__stop_on_max_iterations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            max_iterations = 3

            # Create a simple transformation plan
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(path / "TRANSFORMATION.md")
            )
            transformation_plan.update_iteration(3)

            # Create the agent state
            state: MDEAgentState = {
                "transformation_plan": transformation_plan.to_dict(),
                "latest_evaluation_runs": [],
            }
            result = asyncio.run(
                self.check_transformation_iteration(state, max_iterations)
            )
            self.assertEqual(result, "max_iteration_reached")

    def test_transformation_iteration_control__run_results_have_execution_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            max_iterations = 3

            # Create a simple transformation plan
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(path / "TRANSFORMATION.md")
            )
            transformation_plan.update_iteration(2)

            state: MDEAgentState = {
                "transformation_plan": transformation_plan.to_dict(),
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
            result = asyncio.run(
                self.check_transformation_iteration(state, max_iterations)
            )
            self.assertEqual(result, "error")

    def test_transformation_iteration_control__run_results_have_design_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            max_iterations = 3

            # Create a simple transformation plan
            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(path / "TRANSFORMATION.md")
            )
            transformation_plan.update_iteration(2)

            state: MDEAgentState = {
                "transformation_plan": transformation_plan.to_dict(),
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
            result = asyncio.run(
                self.check_transformation_iteration(state, max_iterations)
            )
            self.assertEqual(result, "design_failed")

    def test_transformation_iteration_control__run_results_no_errors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            max_iterations = 3

            transformation_plan = TransformationPlan.parse(
                FileTransformationPlanParser(path / "TRANSFORMATION.md")
            )
            transformation_plan.update_iteration(2)

            state: MDEAgentState = {
                "transformation_plan": transformation_plan.to_dict(),
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
