import asyncio
import unittest
import datetime

from typing import List

from src.tools.audit.types import AuditRun, AuditResult, Audit, AuditError
from src.tools.audit.executor import AuditExecutor


class MockedAuditCaseImplementation(Audit):
    def __init__(self, audit_id: str, audit_run: AuditRun):
        super().__init__(audit_id)
        self._audit_run = audit_run

    async def setup(self):
        return

    async def run(self):
        return self._audit_run


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
        ]

        executor = AuditExecutor(
            audits=[
                MockedAuditCaseImplementation(
                    audit_id="audit1",
                    audit_run=expected_runs[0],
                ),
                MockedAuditCaseImplementation(
                    audit_id="audit2",
                    audit_run=expected_runs[1],
                ),
            ]
        )

        actual: List[AuditRun] = asyncio.run(executor.execute_all())
        self.assertEqual(len(actual), 2, "Should return runs for both audit cases.")

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

    def test_get_latest_results(self):
        self.assertTrue(
            hasattr(AuditExecutor, "get_latest_results"),
            "AuditExecutor should have a 'get_latest_results' method.",
        )
