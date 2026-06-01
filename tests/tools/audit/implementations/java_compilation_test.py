import asyncio

from typing import List
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

from src.tools.audit.implementations.java_compilation import (
    JavaCompilationAudit,
    parse_javac_output,
    AuditError,
)

ERROR_BLOCK_1 = """
    ublic static void main(String[] args) {
         ^"""
ERROR_BLOCK_2 = """
    String getName() 
                    ^"""
ERROR_BLOCK_3 = """
}
^"""

JAVAC_ERROR_OUTPUT = f"""
./.bx-agent-workspace/test/Family.java:2: Fehler: <ID> erwartet
{ERROR_BLOCK_1}
./.bx-agent-workspace/test/Family.java:6: Fehler: ';' erwartet
{ERROR_BLOCK_2}
./.bx-agent-workspace/test/Family.java:10: Fehler: class, interface, enum oder record erwartet
{ERROR_BLOCK_3}
3 Fehler
"""

JAVAC_SUCCESS_OUTPUT = ""


class TestJavaCompilation(TestCase):

    @patch("shutil.which")
    def test_setup__fail_if_javac_not_found(self, mock_which):
        """
        Test that the setup method fails if javac is not installed on the system.
        """

        self.assertTrue(
            hasattr(JavaCompilationAudit, "setup"),
            "JavaCompilationAudit should have a 'setup' method.",
        )

        mock_which.return_value = None
        java_compilation_audit = JavaCompilationAudit(files=[])

        with self.assertRaises(
            RuntimeError,
            msg="JavaCompilationAudit's 'setup' method should raise RuntimeError if javac is not installed on the system.",
        ):
            asyncio.run(java_compilation_audit.setup())

    def test_execute__method_defined(self):
        self.assertTrue(
            hasattr(JavaCompilationAudit, "run"),
            "JavaCompilationAudit should have a 'run' method.",
        )


class TestJavaCompilationAudit__parse_javac_output(TestCase):

    def test_parse_javac_output__no_errors(self):
        errors = parse_javac_output(JAVAC_SUCCESS_OUTPUT)

        self.assertEqual(
            len(errors),
            0,
            "There should be no AuditErrors for the provided javac success output.",
        )

    def test_parse_javac_output__with_errors(self):
        errors = parse_javac_output(JAVAC_ERROR_OUTPUT)

        self.assertEqual(
            len(errors),
            3,
            "There should be three AuditErrors for the provided javac error output.",
        )

        expected_errors: List[AuditError] = [
            AuditError(
                message="Fehler: <ID> erwartet",
                details={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 2,
                    "block": ERROR_BLOCK_1,
                },
            ),
            AuditError(
                message="Fehler: ';' erwartet",
                details={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 6,
                    "block": ERROR_BLOCK_2,
                },
            ),
            AuditError(
                message="Fehler: class, interface, enum oder record erwartet",
                details={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 10,
                    "block": ERROR_BLOCK_3,
                },
            ),
        ]

        for error, expected_error in zip(errors, expected_errors):
            self.assertEqual(
                expected_error.message,
                error.message,
                f"Expected error message '{expected_error.message}' not found in actual error message '{error.message}'.",
            )
