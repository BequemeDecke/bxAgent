import asyncio
import datetime
import unittest
from dataclasses import asdict
from typing import List

from pydantic import BaseModel

from bxagent.validation.executor import LinkedValidation, ValidationExecutor
from bxagent.validation.types import (
    Validation,
    ValidationError,
    ValidationResult,
    ValidationRun,
)


class MockedValidationCaseImplementation(Validation):
    def __init__(
        self,
        results: List[ValidationResult] = None,
        errors: List[ValidationError] = None,
    ):
        self.results = results or []
        self.errors = errors or []

    async def setup(self):
        return

    async def run(self, **kwargs):
        return (self.results, self.errors)


class MockedValidationSchema(BaseModel):
    param1: str


class FailingValidationCaseImplementation(Validation):
    async def setup(self):
        return

    async def run(self, **kwargs):
        raise Exception("This validation case is designed to fail.")


def assert_validation_run_equal_except(
    equal_method, actual: ValidationRun, expected: ValidationRun
):
    actual_dict = asdict(actual)
    expected_dict = asdict(expected)

    actual_dict.pop("started_at", None)
    expected_dict.pop("started_at", None)
    actual_dict.pop("execution_time_ms", None)
    expected_dict.pop("execution_time_ms", None)

    equal_method(actual_dict, expected_dict)


class TestValidationExecutor__execute_specific(unittest.TestCase):
    def setUp(self):
        self.results = [ValidationResult(content="result1")]
        self.errors = []
        self.executor = ValidationExecutor(
            validations={
                "validation1": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value1"),
                },
                "validation2": {
                    "validation": FailingValidationCaseImplementation(),
                    "validation_schema": MockedValidationSchema(param1="value2"),
                },
            }
        )

    def test_execute_specific__return_validation_runs(self):
        self.assertTrue(
            hasattr(ValidationExecutor, "execute_specific"),
            "ValidationExecutor should have an 'execute_specific' method.",
        )

        expected_run = ValidationRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=self.results,
            errors=self.errors,
        )

        run = asyncio.run(
            self.executor.execute_specific("validation1", input={"param1": "value1"})
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
            "There should be no errors for this validation case.",
        )

    def test_execute_specific__non_existent_validation(self):
        # Edge case: try to execute a non-existent validation and check for ValueError
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for non-existent validation ID."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "non_existent_validation", input={"param1": "value"}
                )
            )

    def test_execute_specific__validation_raises_exception(self):
        # Edge case: execution of an validation that raises an exception should be handled properly
        # It should return an ValidationRun with an appropriate ValidationError instead of propagating the exception
        run = asyncio.run(
            self.executor.execute_specific("validation2", input={"param1": "value2"})
        )
        self.assertEqual(
            len(run.errors),
            1,
            "Should return an validation run with a single error.",
        )
        self.assertEqual(
            run.errors[0].message,
            "This validation case is designed to fail.",
            "Error message should match the expected error message.",
        )

    def test_execute_specific__raises_validation_error(self):
        # Edge case: execution of an validation with invalid input should be handled properly
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for invalid input."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "validation1", input={"invalid_param": "value"}
                )
            )

        with self.assertRaises(
            ValueError, msg="Should raise ValueError for missing required parameter."
        ):
            asyncio.run(self.executor.execute_specific("validation1", input={}))


class TestValidationExecutor__execute_all(unittest.TestCase):
    def setUp(self):
        self.run_1 = ValidationRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[ValidationResult(content="result1")],
            errors=[],
        )
        self.run_2 = ValidationRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[],
            errors=[
                ValidationError(
                    message="error1", details={"exception_type": "Exception"}
                )
            ],
        )
        self.executor = ValidationExecutor(
            validations={
                "validation1": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value1"),
                },
                "validation2": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value2"),
                },
                "validation3": {
                    "validation": FailingValidationCaseImplementation(),
                    "validation_schema": MockedValidationSchema(param1="value3"),
                },
            }
        )

    def test_execute_all__method_defined(self):
        self.assertTrue(
            hasattr(ValidationExecutor, "execute_all"),
            "ValidationExecutor should have an 'execute_all' method.",
        )

    def test_execute_all__return_validation_runs(self):
        expected_runs = [
            ValidationRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=self.run_1.results,
                errors=self.run_1.errors,
            ),
            ValidationRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=self.run_2.results,
                errors=self.run_2.errors,
            ),
            ValidationRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[
                    ValidationError(
                        message="This validation case is designed to fail.",
                        details={"exception_type": "Exception"},
                    )
                ],
            ),
        ]

        actual: List[ValidationRun] = asyncio.run(
            self.executor.execute_all(
                input={
                    "validation1": {"param1": "value1"},
                    "validation2": {"param1": "value2"},
                    "validation3": {"param1": "value3"},
                }
            )
        )
        self.assertEqual(
            len(actual), 3, "Should return runs for all three validation cases."
        )

        for actual_run, expected_run in zip(actual, expected_runs):
            assert_validation_run_equal_except(
                self.assertEqual, actual_run, expected_run
            )


class TestValidationExecutor__get_latest_results(unittest.TestCase):
    def setUp(self):
        self.run_1 = ValidationRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[ValidationResult(content="result1")],
            errors=[],
        )
        self.run_2 = ValidationRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[],
            errors=[
                ValidationError(
                    message="error1", details={"exception_type": "Exception"}
                )
            ],
        )
        self.executor = ValidationExecutor(
            validations={
                "validation1": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value1"),
                },
                "validation2": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value2"),
                },
            }
        )

    def test_get_latest_results__method_defined(self):
        self.assertTrue(
            hasattr(ValidationExecutor, "get_latest_results"),
            "ValidationExecutor should have a 'get_latest_results' method.",
        )

    def test_get_latest_results(self):
        expected_results = [
            self.run_1,
            self.run_2,
        ]

        asyncio.run(
            self.executor.execute_all(
                input={
                    "validation1": {"param1": "value1"},
                    "validation2": {"param1": "value2"},
                    "validation3": {"param1": "value3"},
                }
            )
        )

        latest_results = self.executor.get_latest_results()
        self.assertEqual(
            len(latest_results),
            2,
            "Should return latest results for both validations.",
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
                    "validation1": {"param1": "value1"},
                    "validation2": {"param1": "value2"},
                    "validation3": {"param1": "value3"},
                }
            )
        )

        # Then execute one specific validation again to create a new iteration and check if get_latest_results returns the updated latest result
        asyncio.run(
            self.executor.execute_specific("validation1", input={"param1": "value1"})
        )
        asyncio.run(
            self.executor.execute_specific("validation1", input={"param1": "value1"})
        )

        latest_results = self.executor.get_latest_results()
        for actual_run, expected_run in zip(latest_results, expected_results):
            assert_validation_run_equal_except(
                self.assertEqual, actual_run, expected_run
            )


class TestValidationExecutor__register_linked_validation(unittest.TestCase):
    def setUp(self):
        self.results = [
            ValidationResult(content="result1"),
            ValidationResult(content="result2"),
        ]
        self.errors = [
            ValidationError(message="error", details={"exception_type": "Exception"})
        ]
        self.executor = ValidationExecutor(
            validations={
                "validation1": {
                    "validation": MockedValidationCaseImplementation(),
                    "validation_schema": MockedValidationSchema(param1="value1"),
                },
                "validation2": {
                    "validation": MockedValidationCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "validation_schema": MockedValidationSchema(param1="value2"),
                },
            }
        )

    def test_register_linked_validation__method_defined(self):
        self.assertTrue(
            hasattr(ValidationExecutor, "register_linked_validation"),
            "ValidationExecutor should have a 'register_linked_validation' method.",
        )

    def test_register_linked_validation__link_existing_validation(self):
        self.executor.register_linked_validation("linked_validation", "validation1")
        self.assertIn(
            "linked_validation",
            self.executor.validations,
            "Linked validation should be registered in the executor.",
        )
        self.assertIsInstance(
            self.executor.validations["linked_validation"]["validation"],
            LinkedValidation,
            "Registered linked validation should be an instance of LinkedValidation.",
        )
        self.assertEqual(
            self.executor.iterations["linked_validation"],
            [],
            "Linked validation should have its own iteration list initialized to an empty list.",
        )

    def test_register_linked_validation__non_existent_existing_validation(self):
        with self.assertRaises(
            ValueError,
            msg="Should raise ValueError for non-existent existing validation ID.",
        ):
            self.executor.register_linked_validation(
                "linked_validation", "non_existent_validation"
            )

    def test_register_linked_validation__duplicate_new_validation_id(self):
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for duplicate new validation ID."
        ):
            self.executor.register_linked_validation("validation1", "validation1")

    def test_register_linked_validation__linked_validation_execution(self):
        self.executor.register_linked_validation("linked_validation", "validation2")
        run = asyncio.run(
            self.executor.execute_specific(
                "linked_validation", input={"param1": "value1"}
            )
        )
        self.assertEqual(
            run.results,
            self.results,
            "Linked validation should return the same results as the original validation.",
        )
        self.assertEqual(
            run.errors,
            self.errors,
            "Linked validation should return the same errors as the original validation.",
        )
