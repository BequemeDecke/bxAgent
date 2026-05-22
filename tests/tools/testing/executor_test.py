import asyncio
import unittest
import datetime
from dataclasses import asdict

from typing import List

from src.tools.audit.types import AuditRun, AuditResult, Audit, AuditError
from src.tools.audit.executor import AuditExecutor


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

    async def run(self):
        return (self.results, self.errors)


class FailingAuditCaseImplementation(Audit):
    async def setup(self):
        return

    async def run(self):
        raise Exception("This audit case is designed to fail.")


class TestAuditExecutor(unittest.TestCase):
    def assert_audit_run_equal_except(self, actual: AuditRun, expected: AuditRun):
        actual_dict = asdict(actual)
        expected_dict = asdict(expected)

        actual_dict.pop("started_at", None)
        expected_dict.pop("started_at", None)
        actual_dict.pop("execution_time_ms", None)
        expected_dict.pop("execution_time_ms", None)

        self.assertEqual(actual_dict, expected_dict)

    def test_execute_all(self):
        self.assertTrue(
            hasattr(AuditExecutor, "execute_all"),
            "AuditExecutor should have an 'execute_all' method.",
        )

        expected_runs = [
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[AuditResult(content="result1")],
                errors=[],
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[
                    AuditError(
                        message="error1", details={"exception_type": "Exception"}
                    )
                ],
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

        executor = AuditExecutor(
            audits={
                "audit1": MockedAuditCaseImplementation(
                    results=expected_runs[0].results,
                    errors=expected_runs[0].errors,
                ),
                "audit2": MockedAuditCaseImplementation(
                    results=expected_runs[1].results,
                    errors=expected_runs[1].errors,
                ),
                "audit3": FailingAuditCaseImplementation(),
            }
        )

        actual: List[AuditRun] = asyncio.run(executor.execute_all())
        self.assertEqual(
            len(actual), 3, "Should return runs for all three audit cases."
        )

        for actual_run, expected_run in zip(actual, expected_runs):
            self.assert_audit_run_equal_except(actual_run, expected_run)

    def test_execute_specific(self):
        self.assertTrue(
            hasattr(AuditExecutor, "execute_specific"),
            "AuditExecutor should have an 'execute_specific' method.",
        )

        expected_run = AuditRun(
            started_at=datetime.datetime.now(),
            execution_time_ms=100,
            iteration=1,
            results=[AuditResult(content="result1")],
            errors=[],
        )

        executor = AuditExecutor(
            audits={
                "audit1": MockedAuditCaseImplementation(
                    results=expected_run.results,
                    errors=expected_run.errors,
                ),
                "audit2": FailingAuditCaseImplementation(),
            }
        )

        # Happy path: execute the specific audit and check results
        run = asyncio.run(executor.execute_specific("audit1"))
        self.assertEqual(
            run.results,
            expected_run.results,
            "Results should match the expected results.",
        )

        # Edge case: try to execute a non-existent audit and check for ValueError
        with self.assertRaises(
            ValueError, msg="Should raise ValueError for non-existent audit ID."
        ):
            asyncio.run(executor.execute_specific("non_existent_audit"))

        # Edge case: execution of an audit that raises an exception should be handled properly
        # It should return an AuditRun with an appropriate AuditError instead of propagating the exception
        run = asyncio.run(executor.execute_specific("audit2"))
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

    def test_get_latest_results(self):
        self.assertTrue(
            hasattr(AuditExecutor, "get_latest_results"),
            "AuditExecutor should have a 'get_latest_results' method.",
        )

        expected_results = [
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[AuditResult(content="result1")],
                errors=[],
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[
                    AuditError(
                        message="error1", details={"exception_type": "Exception"}
                    )
                ],
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

        executor = AuditExecutor(
            audits={
                "audit1": MockedAuditCaseImplementation(
                    results=expected_results[0].results,
                    errors=expected_results[0].errors,
                ),
                "audit2": MockedAuditCaseImplementation(
                    results=expected_results[1].results,
                    errors=expected_results[1].errors,
                ),
            }
        )

        # Execute audits to populate iterations
        asyncio.run(executor.execute_all())

        latest_results = executor.get_latest_results()
        self.assertEqual(
            len(latest_results),
            2,
            "Should return latest results for both audits.",
        )

        expected_latest_results = [
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[AuditResult(content="result1")],
                errors=[],
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[
                    AuditError(
                        message="error1", details={"exception_type": "Exception"}
                    )
                ],
            ),
        ]

        for actual_run, expected_run in zip(latest_results, expected_latest_results):
            self.assert_audit_run_equal_except(actual_run, expected_run)

        # Then execute one specific audit again to create a new iteration and check if get_latest_results returns the updated latest result
        asyncio.run(executor.execute_specific("audit1"))
        asyncio.run(executor.execute_specific("audit1"))

        latest_results = executor.get_latest_results()
        self.assertEqual(
            len(latest_results),
            2,
            "Should return latest results for both audits after executing one audit again.",
        )
        self.assertEqual(
            latest_results[0].iteration,
            3,
            "Iteration number for the first audit should be updated to 2 after executing it again.",
        )
