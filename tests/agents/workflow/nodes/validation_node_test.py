"""
This test checks if the evaluation node correctly executes the evaluation core and returns the expected results.

It is more of an integration test that checks the interaction between the evaluation node and the evaluation core, rather than a unit test of the evaluation node itself.
"""

import asyncio
from typing import Any, Dict
from unittest import TestCase
from unittest.mock import Mock

from pydantic import BaseModel

from mdagent.agents.workflow.nodes.evaluation_node import (
    create_evaluation_node,
)
from mdagent.agents.workflow.state import WorkflowState
from mdagent.evaluation import EvaluationExecutor
from mdagent.evaluation.types import Evaluation, EvaluationError, EvaluationResult


class MockedSchema(BaseModel):
    some_field: str


class TestEvaluationNode__ExecutionModeAll(TestCase):
    def test_evaluation_node__updates_state_with_latest_results(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_evaluation = Mock(spec=Evaluation)
        mocked_evaluation.run.return_value = (
            [EvaluationResult(content="Some evaluation result")],
            [EvaluationError(message="Some evaluation error", type="EvaluationError")],
        )

        evaluation_id = "test_evaluation"

        evaluation_executor = EvaluationExecutor(
            evaluations={
                evaluation_id: {
                    "evaluation": mocked_evaluation,
                    "evaluation_schema": MockedSchema,
                }
            }
        )
        evaluation_agent_work = create_evaluation_node(
            evaluation_executor=evaluation_executor,
            mapper={evaluation_id: mocked_mapper},
            execution_mode="all",
        )

        result = asyncio.run(evaluation_agent_work(WorkflowState()))

        self.assertEqual(
            len(result["latest_evaluation_runs"]),
            1,
            "There should be results for one evaluation run.",
        )

        run_result = result["latest_evaluation_runs"][0]
        self.assertEqual(
            len(run_result.results), 1, "There should be one evaluation result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some evaluation result",
            "The evaluation result content should match.",
        )
        self.assertEqual(
            len(run_result.errors), 1, "There should be one evaluation error."
        )
        self.assertEqual(
            run_result.errors[0].message,
            "Some evaluation error",
            "The evaluation error message should match.",
        )

        self.assertTrue(
            mocked_evaluation.run.called,
            "The evaluation's run method should have been called.",
        )
        self.assertEqual(
            mocked_evaluation.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The evaluation should have been called with the mapped parameters.",
        )

    def test_evaluation_node__evaluation_has_no_mapper(self):
        mocked_evaluation = Mock(spec=Evaluation)
        mocked_evaluation.run.return_value = (
            [EvaluationResult(content="Some evaluation result")],
            [EvaluationError(message="Some evaluation error", type="EvaluationError")],
        )

        evaluation_id = "test_evaluation"

        evaluation_executor = EvaluationExecutor(
            evaluations={
                evaluation_id: {
                    "evaluation": mocked_evaluation,
                    "evaluation_schema": MockedSchema,
                }
            }
        )
        evaluation_agent_work = create_evaluation_node(
            evaluation_executor=evaluation_executor,
            mapper={},  # No mapper provided
            execution_mode="all",
        )

        with self.assertRaises(
            KeyError,
            msg="A KeyError should be raised when no mapper is provided for the evaluation.",
        ):
            asyncio.run(evaluation_agent_work(WorkflowState()))


class TestEvaluationNode__ExecutionModeSpecific(TestCase):
    def test_evaluation_node__updates_state_with_latest_results_specific(self):
        def mocked_mapper(state: WorkflowState) -> Dict[str, Any]:
            return {"some_field": "some_value"}

        mocked_evaluation = Mock(spec=Evaluation)
        mocked_evaluation.run.return_value = (
            [EvaluationResult(content="Some evaluation result")],
            [EvaluationError(message="Some evaluation error", type="EvaluationError")],
        )

        evaluation_id = "test_evaluation"

        evaluation_executor = EvaluationExecutor(
            evaluations={
                evaluation_id: {
                    "evaluation": mocked_evaluation,
                    "evaluation_schema": MockedSchema,
                }
            }
        )
        evaluation_agent_work = create_evaluation_node(
            evaluation_executor=evaluation_executor,
            mapper={evaluation_id: mocked_mapper},
            execution_mode="specific",
        )

        result = asyncio.run(evaluation_agent_work(WorkflowState()))

        self.assertIn(
            evaluation_id,
            result["latest_evaluation_runs"],
            "The latest evaluation runs should contain the specific evaluation id.",
        )

        run_result = result["latest_evaluation_runs"][evaluation_id]
        self.assertEqual(
            len(run_result.results), 1, "There should be one evaluation result."
        )
        self.assertEqual(
            run_result.results[0].content,
            "Some evaluation result",
            "The evaluation result content should match.",
        )
        self.assertEqual(
            len(run_result.errors), 1, "There should be one evaluation error."
        )
        self.assertEqual(
            run_result.errors[0].message,
            "Some evaluation error",
            "The evaluation error message should match.",
        )

        self.assertTrue(
            mocked_evaluation.run.called,
            "The evaluation's run method should have been called.",
        )
        self.assertEqual(
            mocked_evaluation.run.call_args.kwargs,
            {"some_field": "some_value"},
            "The evaluation should have been called with the mapped parameters.",
        )

    def test_evaluation_node__evaluation_has_no_mapper_specific(self):
        mocked_evaluation = Mock(spec=Evaluation)
        mocked_evaluation.run.return_value = (
            [EvaluationResult(content="Some evaluation result")],
            [EvaluationError(message="Some evaluation error", type="EvaluationError")],
        )

        evaluation_id = "test_evaluation"

        evaluation_executor = EvaluationExecutor(
            evaluations={
                evaluation_id: {
                    "evaluation": mocked_evaluation,
                    "evaluation_schema": MockedSchema,
                }
            }
        )
        evaluation_agent_work = create_evaluation_node(
            evaluation_executor=evaluation_executor,
            mapper={},  # No mapper provided
            execution_mode="specific",
        )

        self.assertEqual(
            asyncio.run(evaluation_agent_work(WorkflowState())),
            {"latest_evaluation_runs": {}},
            "When no mapper is provided for specific execution mode, the latest evaluation runs should be empty.",
        )
