import asyncio
import unittest
import datetime

from typing import List

from src.tools.audit.types import AuditRun, AuditResult, Audit, AuditError
from src.tools.audit.executor import AuditExecutor


class MockedAuditCaseImplementation(Audit):
    def __init__(self, audit_id: str, results: List[AuditResult] = None, errors: List[AuditError] = None):
        super().__init__(audit_id)
        self.results = results or []
        self.errors = errors or []

    async def setup(self):
        return

    async def run(self):
        return (self.results, self.errors)
    

class FailingAuditCaseImplementation(Audit):
    def __init__(self, audit_id: str):
        super().__init__(audit_id)

    async def setup(self):
        return

    async def run(self):
        raise Exception("This audit case is designed to fail.")


class TestAuditExecutor(unittest.TestCase):
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
                errors=[AuditError(message="error1")],
            ),
            AuditRun(
                started_at=datetime.datetime.now(),
                execution_time_ms=100,
                iteration=1,
                results=[],
                errors=[AuditError(message="This audit case is designed to fail.")],
            ),
        ]

        executor = AuditExecutor(
            audits=[
                MockedAuditCaseImplementation(
                    audit_id="audit1",
                    results=expected_runs[0].results,
                    errors=expected_runs[0].errors,
                ),
                MockedAuditCaseImplementation(
                    audit_id="audit2",
                    results=expected_runs[1].results,
                    errors=expected_runs[1].errors,
                ),
                FailingAuditCaseImplementation(
                    audit_id="audit3",
                ),
            ]
        )

        actual: List[AuditRun] = asyncio.run(executor.execute_all())
        self.assertEqual(len(actual), 3, "Should return runs for all three audit cases.")

        for actual_run, expected_run in zip(actual, expected_runs):
            self.assertEqual(
                actual_run.results,
                expected_run.results,
                "Results should match the expected results.",
            )
            self.assertEqual(
                actual_run.errors,
                expected_run.errors,
                "Errors should match the expected errors.",
            )

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
            audits=[
                MockedAuditCaseImplementation(
                    audit_id="audit1",
                    results=expected_run.results,
                    errors=expected_run.errors,
                ),
                FailingAuditCaseImplementation(
                    audit_id="audit2",
                ),
            ]
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
