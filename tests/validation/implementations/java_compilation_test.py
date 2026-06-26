import asyncio
import subprocess
from pathlib import Path
from typing import List
from unittest import TestCase
from unittest.mock import patch

from bxagent.validation.implementations.java_compilation import (
    JavaCompilationValidation,
    ValidationError,
    parse_javac_output,
)
from bxagent.validation.types import ValidationResult

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

SYMBOL_ERRORS = """
./.bx-agent-workspace/transformation/transformation/Family.java:29: Fehler: Symbol nicht gefunden
    FamilyMember getFather();
    ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./.bx-agent-workspace/transformation/transformation/Family.java:36: Fehler: Symbol nicht gefunden
    void setFather(FamilyMember father);
                   ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./.bx-agent-workspace/transformation/transformation/Family.java:43: Fehler: Symbol nicht gefunden
    FamilyMember getMother();
    ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./.bx-agent-workspace/transformation/transformation/Family.java:50: Fehler: Symbol nicht gefunden
    void setMother(FamilyMember mother);
                   ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./.bx-agent-workspace/transformation/transformation/Family.java:57: Fehler: Symbol nicht gefunden
    List<FamilyMember> getSons();
         ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
./.bx-agent-workspace/transformation/transformation/Family.java:64: Fehler: Symbol nicht gefunden
    List<FamilyMember> getDaughters();
         ^
  Symbol: Klasse FamilyMember
  Ort: Schnittstelle Family
6 Fehler"""


class TestJavaCompilation(TestCase):
    @patch("shutil.which")
    def test_setup__fail_if_javac_not_found(self, mock_which):
        """
        Test that the setup method fails if javac is not installed on the system.
        """

        self.assertTrue(
            hasattr(JavaCompilationValidation, "setup"),
            "JavaCompilationValidation should have a 'setup' method.",
        )

        mock_which.return_value = None
        java_compilation_validation = JavaCompilationValidation()

        with self.assertRaises(
            RuntimeError,
            msg="JavaCompilationValidation's 'setup' method should raise RuntimeError if javac is not installed on the system.",
        ):
            asyncio.run(java_compilation_validation.setup())

    def test_run__method_defined(self):
        self.assertTrue(
            hasattr(JavaCompilationValidation, "run"),
            "JavaCompilationValidation should have a 'run' method.",
        )

    @patch("subprocess.run")
    def test_run__invalid_syntax(self, mock_subprocess_run):
        """
        Test that the run method returns an ValidationError when javac returns a syntax error.
        """

        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["javac", "Test.java"],
            returncode=1,
            stdout="",
            stderr=JAVAC_ERROR_OUTPUT,
        )

        java_compilation_validation = JavaCompilationValidation()
        results, errors = asyncio.run(
            java_compilation_validation.run(files=[Path("Test.java")])
        )

        self.assertEqual(
            len(results),
            3,
            "There should be three failed ValidationResults when javac returns a syntax error.",
        )
        self.assertEqual(
            len(errors),
            0,
            "There should be no ValidationErrors for the provided javac error output.",
        )

        for result in results:
            self.assertIn(
                result.content,
                [
                    "Fehler: <ID> erwartet",
                    "Fehler: ';' erwartet",
                    "Fehler: class, interface, enum oder record erwartet",
                ],
                f"Expected error message not found in actual error message '{result.content}'.",
            )

    @patch("subprocess.run")
    def test_run__valid_syntax(self, mock_subprocess_run):
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["javac", "Test.java"],
            returncode=0,
            stdout=JAVAC_SUCCESS_OUTPUT,
            stderr="",
        )

        java_compilation_validation = JavaCompilationValidation()
        results, errors = asyncio.run(
            java_compilation_validation.run(files=[Path("Test.java")])
        )

        self.assertEqual(
            len(results),
            1,
            "There should be one ValidationResult when javac returns a success output (since we haven't implemented result parsing yet).",
        )
        self.assertEqual(
            len(errors),
            0,
            "There should be no ValidationErrors when javac returns a success output.",
        )

    def test_run__missing_files_parameter(self):
        java_compilation_validation = JavaCompilationValidation()

        with self.assertRaises(
            ValueError,
            msg="JavaCompilationValidation's 'run' method should raise ValueError if 'files' parameter is missing.",
        ):
            asyncio.run(java_compilation_validation.run())

    @patch("subprocess.run")
    def test_run__exception_occurs_during_javac(self, mock_subprocess_run):
        mock_subprocess_run.side_effect = Exception("Test exception")

        java_compilation_validation = JavaCompilationValidation()
        results, errors = asyncio.run(
            java_compilation_validation.run(files=[Path("Test.java")])
        )

        self.assertEqual(
            len(results),
            0,
            "There should be no ValidationResult objects when an exception occurs during javac execution.",
        )
        self.assertEqual(
            len(errors),
            1,
            "There should be one ValidationError when an exception occurs during javac execution.",
        )


class TestJavaCompilationValidation__parse_javac_output(TestCase):
    def test_parse_javac_output__no_errors(self):
        errors = parse_javac_output(JAVAC_SUCCESS_OUTPUT)

        self.assertEqual(
            len(errors),
            0,
            "There should be no ValidationErrors for the provided javac success output.",
        )

    def test_parse_javac_output__with_errors(self):
        errors = parse_javac_output(JAVAC_ERROR_OUTPUT)

        self.assertEqual(
            len(errors),
            3,
            "There should be three ValidationResult objects for the provided javac error output.",
        )

        expected_errors: List[ValidationResult] = [
            ValidationResult(
                content="Fehler: <ID> erwartet",
                metadata={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 2,
                    "block": ERROR_BLOCK_1,
                },
            ),
            ValidationResult(
                content="Fehler: ';' erwartet",
                metadata={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 6,
                    "block": ERROR_BLOCK_2,
                },
            ),
            ValidationResult(
                content="Fehler: class, interface, enum oder record erwartet",
                metadata={
                    "file": "./.bx-agent-workspace/test/Family.java",
                    "line": 10,
                    "block": ERROR_BLOCK_3,
                },
            ),
        ]

        for error, expected_error in zip(errors, expected_errors):
            self.assertEqual(
                expected_error.content,
                error.content,
                f"Expected error message '{expected_error.content}' not found in actual error message '{error.content}'.",
            )

    def test_parse_javac_output__symbol_errors(self):
        errors = parse_javac_output(SYMBOL_ERRORS)

        self.assertEqual(
            len(errors),
            6,
            "There should be six ValidationResult objects for the provided javac symbol error output.",
        )

        expected_error_messages = [
            "Fehler: Symbol nicht gefunden",
        ]

        for error in errors:
            self.assertIn(
                error.content,
                expected_error_messages,
                f"Expected error message not found in actual error message '{error.content}'.",
            )
