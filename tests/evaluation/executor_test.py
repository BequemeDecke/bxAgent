import asyncio
import unittest
from dataclasses import asdict
from datetime import UTC, datetime

from pydantic import BaseModel

from mdeagent.evaluation.executor import EvaluationExecutor, LinkedEvaluation
from mdeagent.evaluation.types import (
    Evaluation,
    EvaluationError,
    EvaluationResult,
    EvaluationRun,
)


class MockedEvaluationCaseImplementation(Evaluation):
    def __init__(
        self,
        results: list[EvaluationResult] | None = None,
        errors: list[EvaluationError] | None = None,
    ):
        self.results = results or []
        self.errors = errors or []

    async def setup(self):
        return

    async def run(self, **kwargs):
        return (self.results, self.errors)


class MockedEvaluationSchema(BaseModel):
    param1: str


class FailingEvaluationCaseImplementation(Evaluation):
    async def setup(self):
        return

    async def run(self, **kwargs):
        raise Exception("This evaluation case is designed to fail.")


class FailingSetupEvaluationCaseImplementation(Evaluation):
    async def setup(self):
        raise Exception("This evaluation setup is designed to fail.")

    async def run(self, **kwargs):
        return ([], [])


def assert_evaluation_run_equal_except(
    equal_method, actual: EvaluationRun, expected: EvaluationRun
):
    actual_dict = asdict(actual)
    expected_dict = asdict(expected)

    actual_dict.pop("started_at", None)
    expected_dict.pop("started_at", None)
    actual_dict.pop("execution_time_ms", None)
    expected_dict.pop("execution_time_ms", None)

    equal_method(actual_dict, expected_dict)


class TestEvaluationExecutor__execute_specific(unittest.TestCase):
    def setUp(self):
        self.results = [EvaluationResult(content="result1")]
        self.errors = []
        self.executor = EvaluationExecutor(
            evaluations={
                "evaluation1": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value1"),
                    "category": "execution"
                },
                "evaluation2": {
                    "evaluation": FailingEvaluationCaseImplementation(),
                    "evaluation_schema": MockedEvaluationSchema(param1="value2"),
                    "category": "execution"
                },
            }
        )

    def test_execute_specific__return_evaluation_runs(self):
        self.assertTrue(
            hasattr(EvaluationExecutor, "execute_specific"),
            "EvaluationExecutor should have an 'execute_specific' method.",
        )

        expected_run = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=self.results,
            errors=self.errors,
        )

        run = asyncio.run(
            self.executor.execute_specific("evaluation1", input={"param1": "value1"})
        )
        self.assertEqual(
            run.results,
            expected_run.results,
            "Results should match the expected results.",
        )
        self.assertEqual(
            run.iteration,
            expected_run.iteration,
            "Iteration number should match the expected iteration number.",
        )
        self.assertEqual(
            len(run.errors),
            0,
            "There should be no errors for this evaluation case.",
        )
        self.assertEqual(
            run.category,
            expected_run.category,
            "Category should match the expected category.",
        )

    def test_execute_specific__non_existent_evaluation(self):
        # Edge case: try to execute a non-existent evaluation and check for ValueError
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for non-existent evaluation ID."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "non_existent_evaluation", input={"param1": "value"}
                )
            )

    def test_execute_specific__evaluation_raises_exception(self):
        # Edge case: execution of an evaluation that raises an exception should be handled properly
        # It should return an EvaluationRun with an appropriate EvaluationError instead of propagating the exception
        run = asyncio.run(
            self.executor.execute_specific("evaluation2", input={"param1": "value2"})
        )
        self.assertEqual(
            len(run.errors),
            1,
            "Should return an evaluation run with a single error.",
        )
        self.assertEqual(
            run.errors[0].message,
            "This evaluation case is designed to fail.",
            "Error message should match the expected error message.",
        )

    def test_execute_specific__raises_evaluation_error(self):
        # Edge case: execution of an evaluation with invalid input should be handled properly
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for invalid input."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "evaluation1", input={"invalid_param": "value"}
                )
            )

        with self.assertRaises(
            ValueError, msg="Should raise ValueError for missing required parameter."
        ):
            asyncio.run(self.executor.execute_specific("evaluation1", input={}))


class TestEvaluationExecutor__execute_all(unittest.TestCase):
    def setUp(self):
        self.run_1 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            results=[EvaluationResult(content="result1")],
            errors=[],
        )
        self.run_2 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            results=[],
            errors=[
                EvaluationError(
                    message="error1",
                    type="EvaluationError",
                    details={"exception_type": "Exception"},
                )
            ],
        )
        self.executor = EvaluationExecutor(
            evaluations={
                "evaluation1": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value1"),
                    "category": "execution"
                },
                "evaluation2": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value2"),
                    "category": "execution"
                },
                "evaluation3": {
                    "evaluation": FailingEvaluationCaseImplementation(),
                    "evaluation_schema": MockedEvaluationSchema(param1="value3"),
                    "category": "execution"
                },
            }
        )

    def test_execute_all__method_defined(self):
        self.assertTrue(
            hasattr(EvaluationExecutor, "execute_all"),
            "EvaluationExecutor should have an 'execute_all' method.",
        )

    def test_execute_all__return_evaluation_runs(self):
        expected_runs: dict[str, EvaluationRun] = {
            "evaluation1": EvaluationRun(
                started_at=datetime.now(tz=UTC),
                execution_time_ms=100,
                iteration=1,
                category="execution",
                results=self.run_1.results,
                errors=self.run_1.errors,
            ),
            "evaluation2": EvaluationRun(
                started_at=datetime.now(tz=UTC),
                execution_time_ms=100,
                iteration=1,
                category="execution",
                results=self.run_2.results,
                errors=self.run_2.errors,
            ),
            "evaluation3": EvaluationRun(
                started_at=datetime.now(tz=UTC),
                execution_time_ms=100,
                iteration=1,
                category="execution",
                results=[],
                errors=[
                    EvaluationError(
                        message="This evaluation case is designed to fail.",
                        type="Exception",
                        details={"exception_type": "Exception"},
                    )
                ],
            ),
        }

        actual: dict[str, EvaluationRun] = asyncio.run(
            self.executor.execute_all(
                input={
                    "evaluation1": {"param1": "value1"},
                    "evaluation2": {"param1": "value2"},
                    "evaluation3": {"param1": "value3"},
                }
            )
        )
        self.assertEqual(
            len(actual), 3, "Should return runs for all three evaluation cases."
        )
        self.assertIn("evaluation1", actual)
        self.assertIn("evaluation2", actual)
        self.assertIn("evaluation3", actual)

        assert_evaluation_run_equal_except(
            self.assertEqual, actual["evaluation1"], expected_runs["evaluation1"]
        )
        assert_evaluation_run_equal_except(
            self.assertEqual, actual["evaluation2"], expected_runs["evaluation2"]
        )
        assert_evaluation_run_equal_except(
            self.assertEqual, actual["evaluation3"], expected_runs["evaluation3"]
        )

    def test_execute_all__ignores_linked_evaluations(self):
        # Register a linked evaluation
        self.executor.register_linked_evaluation("linked_eval", "evaluation1")
        
        # Execute all - should only execute non-linked evaluations (3 runs, not 4)
        actual: dict[str, EvaluationRun] = asyncio.run(
            self.executor.execute_all(
                input={
                    "evaluation1": {"param1": "value1"},
                    "evaluation2": {"param1": "value2"},
                    "evaluation3": {"param1": "value3"},
                    # Note: No input for "linked_eval" - it should be ignored
                }
            )
        )
        
        self.assertEqual(
            len(actual), 3, 
            "Should return runs for only non-linked evaluations (linked_eval should be ignored)."
        )
        
        # Verify that iteration lists were created only for non-linked evaluations
        self.assertEqual(
            len(self.executor.iterations["evaluation1"]), 1,
            "Non-linked evaluation should have one iteration."
        )
        self.assertEqual(
            len(self.executor.iterations["linked_eval"]), 0,
            "Linked evaluation should have no iterations after execute_all."
        )
        # Verify linked_eval key is NOT in the result dict
        self.assertNotIn("linked_eval", actual)


class TestEvaluationExecutor__get_latest_results(unittest.TestCase):
    def setUp(self):
        self.run_1 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            category="execution",
            iteration=1,
            results=[EvaluationResult(content="result1")],
            errors=[],
        )
        self.run_2 = EvaluationRun(
            started_at=datetime.now(tz=UTC),
            execution_time_ms=100,
            iteration=1,
            category="execution",
            results=[],
            errors=[
                EvaluationError(
                    message="error1",
                    type="EvaluationError",
                    details={"exception_type": "Exception"},
                )
            ],
        )
        self.executor = EvaluationExecutor(
            evaluations={
                "evaluation1": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value1"),
                    "category": "execution",
                },
                "evaluation2": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value2"),
                    "category": "execution",
                },
            }
        )

    def test_get_latest_results__method_defined(self):
        self.assertTrue(
            hasattr(EvaluationExecutor, "get_latest_results"),
            "EvaluationExecutor should have a 'get_latest_results' method.",
        )

    def test_get_latest_results(self):
        expected_results = [
            self.run_1,
            self.run_2,
        ]

        asyncio.run(
            self.executor.execute_all(
                input={
                    "evaluation1": {"param1": "value1"},
                    "evaluation2": {"param1": "value2"},
                    "evaluation3": {"param1": "value3"},
                }
            )
        )

        latest_results = self.executor.get_latest_results()
        self.assertEqual(
            len(latest_results),
            2,
            "Should return latest results for both evaluations.",
        )

    def test_get_latest_results__multiple_iterations(self):
        self.run_1.iteration = 3

        expected_results = [
            self.run_1,
            self.run_2,
        ]
        asyncio.run(
            self.executor.execute_all(
                input={
                    "evaluation1": {"param1": "value1"},
                    "evaluation2": {"param1": "value2"},
                    "evaluation3": {"param1": "value3"},
                }
            )
        )

        # Then execute one specific evaluation again to create a new iteration and check if get_latest_results returns the updated latest result
        asyncio.run(
            self.executor.execute_specific("evaluation1", input={"param1": "value1"})
        )
        asyncio.run(
            self.executor.execute_specific("evaluation1", input={"param1": "value1"})
        )

        latest_results = self.executor.get_latest_results()
        for actual_run, expected_run in zip(latest_results, expected_results):
            assert_evaluation_run_equal_except(
                self.assertEqual, actual_run, expected_run
            )


class TestEvaluationExecutor__register_linked_evaluation(unittest.TestCase):
    def setUp(self):
        self.results = [
            EvaluationResult(content="result1"),
            EvaluationResult(content="result2"),
        ]
        self.errors = [
            EvaluationError(
                message="error",
                type="EvaluationError",
                details={"exception_type": "Exception"},
            )
        ]
        self.executor = EvaluationExecutor(
            evaluations={
                "evaluation1": {
                    "evaluation": MockedEvaluationCaseImplementation(),
                    "evaluation_schema": MockedEvaluationSchema(param1="value1"),
                    "category": "execution"
                },
                "evaluation2": {
                    "evaluation": MockedEvaluationCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "evaluation_schema": MockedEvaluationSchema(param1="value2"),
                    "category": "execution"
                },
            }
        )

    def test_register_linked_evaluation__method_defined(self):
        self.assertTrue(
            hasattr(EvaluationExecutor, "register_linked_evaluation"),
            "EvaluationExecutor should have a 'register_linked_evaluation' method.",
        )

    def test_register_linked_evaluation__link_existing_evaluation(self):
        self.executor.register_linked_evaluation("linked_evaluation", "evaluation1")
        self.assertIn(
            "linked_evaluation",
            self.executor.evaluations,
            "Linked evaluation should be registered in the executor.",
        )
        self.assertIsInstance(
            self.executor.evaluations["linked_evaluation"]["evaluation"],
            LinkedEvaluation,
            "Registered linked evaluation should be an instance of LinkedEvaluation.",
        )
        self.assertEqual(
            self.executor.iterations["linked_evaluation"],
            [],
            "Linked evaluation should have its own iteration list initialized to an empty list.",
        )
        self.assertEqual(
            self.executor.evaluations["linked_evaluation"]["category"],
            self.executor.evaluations["evaluation1"]["category"],
            "Linked evaluation should inherit the category from the existing evaluation.",
        )

    def test_register_linked_evaluation__non_existent_existing_evaluation(self):
        with self.assertRaises(
            ValueError,
            msg="Should raise ValueError for non-existent existing evaluation ID.",
        ):
            self.executor.register_linked_evaluation(
                "linked_evaluation", "non_existent_evaluation"
            )

    def test_register_linked_evaluation__duplicate_new_evaluation_id(self):
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for duplicate new evaluation ID."
        ):
            self.executor.register_linked_evaluation("evaluation1", "evaluation1")

    def test_register_linked_evaluation__linked_evaluation_execution(self):
        self.executor.register_linked_evaluation("linked_evaluation", "evaluation2")
        run = asyncio.run(
            self.executor.execute_specific(
                "linked_evaluation", input={"param1": "value1"}
            )
        )
        self.assertEqual(
            run.results,
            self.results,
            "Linked evaluation should return the same results as the original evaluation.",
        )
        self.assertEqual(
            run.errors,
            self.errors,
            "Linked evaluation should return the same errors as the original evaluation.",
        )


class TestEvaluationExecutor__setup_error_handling(unittest.TestCase):
    def setUp(self):
        self.executor = EvaluationExecutor(
            evaluations={
                "failing_setup": {
                    "evaluation": FailingSetupEvaluationCaseImplementation(),
                    "evaluation_schema": MockedEvaluationSchema(param1="value1"),
                    "category": "setup_failure"
                },
            }
        )

    def test_execute_specific__setup_raises_exception(self):
        # Edge case: execution of an evaluation where setup() raises an exception
        # Should return an EvaluationRun with an appropriate EvaluationError instead of propagating the exception
        run = asyncio.run(
            self.executor.execute_specific("failing_setup", input={"param1": "value1"})
        )
        self.assertEqual(
            len(run.errors),
            1,
            "Should return an evaluation run with a single error.",
        )
        self.assertEqual(
            run.errors[0].message,
            "This evaluation setup is designed to fail.",
            "Error message should match the expected error message.",
        )
        self.assertEqual(
            run.errors[0].type,
            "Exception",
            "Error type should be 'Exception'.",
        )
        self.assertEqual(
            run.iteration,
            1,
            "Iteration should be 1 even when setup fails.",
        )
        self.assertEqual(
            len(run.results),
            0,
            "Should have no results when setup fails.",
        )
