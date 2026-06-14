import asyncio
import unittest
import datetime

from dataclasses import asdict
from typing import List
from pydantic import BaseModel

from bxagent.tools.audit.types import AuditRun, AuditResult, Audit, AuditError
from bxagent.tools.audit.executor import AuditExecutor, LinkedAudit


class MockedAuditCaseImplementation(Audit):
    def __init__(
        self,
        results: List[AuditResult] = None,
        errors: List[AuditError] = None,
    ):
        self.results = results or []
        self.errors = errors or []

    async def setup(self):
        return

    async def run(self, **kwargs):
        return (self.results, self.errors)


class MockedAuditSchema(BaseModel):
    param1: str


class FailingAuditCaseImplementation(Audit):
    async def setup(self):
        return

    async def run(self, **kwargs):
        raise Exception("This audit case is designed to fail.")


def assert_audit_run_equal_except(equal_method, actual: AuditRun, expected: AuditRun):
    actual_dict = asdict(actual)
    expected_dict = asdict(expected)

    actual_dict.pop("started_at", None)
    expected_dict.pop("started_at", None)
    actual_dict.pop("execution_time_ms", None)
    expected_dict.pop("execution_time_ms", None)

    equal_method(actual_dict, expected_dict)


class TestAuditExecutor__execute_specific(unittest.TestCase):
    def setUp(self):
        self.results = [AuditResult(content="result1")]
        self.errors = []
        self.executor = AuditExecutor(
            audits={
                "audit1": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value1"),
                },
                "audit2": {
                    "audit": FailingAuditCaseImplementation(),
                    "audit_schema": MockedAuditSchema(param1="value2"),
                },
            }
        )

    def test_execute_specific__return_audit_runs(self):
        self.assertTrue(
            hasattr(AuditExecutor, "execute_specific"),
            "AuditExecutor should have an 'execute_specific' method.",
        )

        expected_run = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=self.results,
            errors=self.errors,
        )

        run = asyncio.run(
            self.executor.execute_specific("audit1", input={"param1": "value1"})
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
            "There should be no errors for this audit case.",
        )

    def test_execute_specific__non_existent_audit(self):
        # Edge case: try to execute a non-existent audit and check for ValueError
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for non-existent audit ID."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "non_existent_audit", input={"param1": "value"}
                )
            )

    def test_execute_specific__audit_raises_exception(self):
        # Edge case: execution of an audit that raises an exception should be handled properly
        # It should return an AuditRun with an appropriate AuditError instead of propagating the exception
        run = asyncio.run(
            self.executor.execute_specific("audit2", input={"param1": "value2"})
        )
        self.assertEqual(
            len(run.errors),
            1,
            "Should return an audit run with a single error.",
        )
        self.assertEqual(
            run.errors[0].message,
            "This audit case is designed to fail.",
            "Error message should match the expected error message.",
        )

    def test_execute_specific__raises_validation_error(self):
        # Edge case: execution of an audit with invalid input should be handled properly
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for invalid input."
        ):
            asyncio.run(
                self.executor.execute_specific(
                    "audit1", input={"invalid_param": "value"}
                )
            )

        with self.assertRaises(
            ValueError, msg="Should raise ValueError for missing required parameter."
        ):
            asyncio.run(self.executor.execute_specific("audit1", input={}))


class TestAuditExecutor__execute_all(unittest.TestCase):
    def setUp(self):
        self.run_1 = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[AuditResult(content="result1")],
            errors=[],
        )
        self.run_2 = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[],
            errors=[
                AuditError(message="error1", details={"exception_type": "Exception"})
            ],
        )
        self.executor = AuditExecutor(
            audits={
                "audit1": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value1"),
                },
                "audit2": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value2"),
                },
                "audit3": {
                    "audit": FailingAuditCaseImplementation(),
                    "audit_schema": MockedAuditSchema(param1="value3"),
                },
            }
        )

    def test_execute_all__method_defined(self):
        self.assertTrue(
            hasattr(AuditExecutor, "execute_all"),
            "AuditExecutor should have an 'execute_all' method.",
        )

    def test_execute_all__return_audit_runs(self):
        expected_runs = [
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=self.run_1.results,
                errors=self.run_1.errors,
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=self.run_2.results,
                errors=self.run_2.errors,
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[
                    AuditError(
                        message="This audit case is designed to fail.",
                        details={"exception_type": "Exception"},
                    )
                ],
            ),
        ]

        actual: List[AuditRun] = asyncio.run(
            self.executor.execute_all(
                input={
                    "audit1": {"param1": "value1"},
                    "audit2": {"param1": "value2"},
                    "audit3": {"param1": "value3"},
                }
            )
        )
        self.assertEqual(
            len(actual), 3, "Should return runs for all three audit cases."
        )

        for actual_run, expected_run in zip(actual, expected_runs):
            assert_audit_run_equal_except(self.assertEqual, actual_run, expected_run)


class TestAuditExecutor__get_latest_results(unittest.TestCase):
    def setUp(self):
        self.run_1 = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[AuditResult(content="result1")],
            errors=[],
        )
        self.run_2 = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[],
            errors=[
                AuditError(message="error1", details={"exception_type": "Exception"})
            ],
        )
        self.executor = AuditExecutor(
            audits={
                "audit1": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.run_1.results,
                        errors=self.run_1.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value1"),
                },
                "audit2": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.run_2.results,
                        errors=self.run_2.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value2"),
                },
            }
        )

    def test_get_latest_results__method_defined(self):
        self.assertTrue(
            hasattr(AuditExecutor, "get_latest_results"),
            "AuditExecutor should have a 'get_latest_results' method.",
        )

    def test_get_latest_results(self):
        expected_results = [
            self.run_1,
            self.run_2,
        ]

        asyncio.run(
            self.executor.execute_all(
                input={
                    "audit1": {"param1": "value1"},
                    "audit2": {"param1": "value2"},
                    "audit3": {"param1": "value3"},
                }
            )
        )

        latest_results = self.executor.get_latest_results()
        self.assertEqual(
            len(latest_results),
            2,
            "Should return latest results for both audits.",
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
                    "audit1": {"param1": "value1"},
                    "audit2": {"param1": "value2"},
                    "audit3": {"param1": "value3"},
                }
            )
        )

        # Then execute one specific audit again to create a new iteration and check if get_latest_results returns the updated latest result
        asyncio.run(
            self.executor.execute_specific("audit1", input={"param1": "value1"})
        )
        asyncio.run(
            self.executor.execute_specific("audit1", input={"param1": "value1"})
        )

        latest_results = self.executor.get_latest_results()
        for actual_run, expected_run in zip(latest_results, expected_results):
            assert_audit_run_equal_except(self.assertEqual, actual_run, expected_run)


class TestAuditExecutor__register_linked_audit(unittest.TestCase):
    def setUp(self):
        self.results = [AuditResult(content="result1"), AuditResult(content="result2")]
        self.errors = [
            AuditError(message="error", details={"exception_type": "Exception"})
        ]
        self.executor = AuditExecutor(
            audits={
                "audit1": {
                    "audit": MockedAuditCaseImplementation(),
                    "audit_schema": MockedAuditSchema(param1="value1"),
                },
                "audit2": {
                    "audit": MockedAuditCaseImplementation(
                        results=self.results,
                        errors=self.errors,
                    ),
                    "audit_schema": MockedAuditSchema(param1="value2"),
                },
            }
        )

    def test_register_linked_audit__method_defined(self):
        self.assertTrue(
            hasattr(AuditExecutor, "register_linked_audit"),
            "AuditExecutor should have a 'register_linked_audit' method.",
        )

    def test_register_linked_audit__link_existing_audit(self):
        self.executor.register_linked_audit("linked_audit", "audit1")
        self.assertIn(
            "linked_audit",
            self.executor.audits,
            "Linked audit should be registered in the executor.",
        )
        self.assertIsInstance(
            self.executor.audits["linked_audit"]["audit"],
            LinkedAudit,
            "Registered linked audit should be an instance of LinkedAudit.",
        )
        self.assertEqual(
            self.executor.iterations["linked_audit"],
            [],
            "Linked audit should have its own iteration list initialized to an empty list.",
        )

    def test_register_linked_audit__non_existent_existing_audit(self):
        with self.assertRaises(
            ValueError,
            msg="Should raise ValueError for non-existent existing audit ID.",
        ):
            self.executor.register_linked_audit("linked_audit", "non_existent_audit")

    def test_register_linked_audit__duplicate_new_audit_id(self):
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for duplicate new audit ID."
        ):
            self.executor.register_linked_audit("audit1", "audit1")

    def test_register_linked_audit__linked_audit_execution(self):
        self.executor.register_linked_audit("linked_audit", "audit2")
        run = asyncio.run(
            self.executor.execute_specific("linked_audit", input={"param1": "value1"})
        )
        self.assertEqual(
            run.results,
            self.results,
            "Linked audit should return the same results as the original audit.",
        )
        self.assertEqual(
            run.errors,
            self.errors,
            "Linked audit should return the same errors as the original audit.",
        )
