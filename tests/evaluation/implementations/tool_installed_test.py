import asyncio
from unittest import TestCase
from unittest.mock import patch

from mdeagent.evaluation.implementations.tool_installed import (
    ToolInstalledEvaluation,
)


class ToolInstalled(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(ToolInstalledEvaluation, "setup"),
            "ToolInstalledEvaluation should have a 'setup' method.",
        )

        tool_installed_evaluation = ToolInstalledEvaluation()

        self.assertIsNone(
            asyncio.run(tool_installed_evaluation.setup()),
            "ToolInstalledEvaluation's 'setup' method should return None.",
        )

    @patch("shutil.which")
    def test_run__check_tools(self, mock_which):
        mock_which.side_effect = lambda tool: (
            "/usr/bin/python" if tool == "python" else None
        )
        tool_installed_evaluation = ToolInstalledEvaluation()

        results, errors = asyncio.run(
            tool_installed_evaluation.run(tools=["python", "nonexistenttool"])
        )

        self.assertEqual(
            len(results),
            2,
            "There should be two results for the evaluation.",
        )
        self.assertEqual(
            len(errors),
            0,
            "There should be no errors occurred during evaluation.",
        )
        self.assertIn(
            "Command 'python' is installed on the system.",
            [result.content for result in results],
            "Expected success message for 'python' was not returned.",
        )
        self.assertTrue(
            any(
                result.metadata.get("success") is True
                for result in results
                if "python" in result.content
            ),
            "Expected success message for 'python' was not returned.",
        )
        self.assertTrue(
            any(
                result.metadata.get("success") is False
                for result in results
                if "nonexistenttool" in result.content
            )
        )

    @patch("shutil.which")
    def test_run__exception_in_which(self, mock_which):
        mock_which.side_effect = Exception("Unexpected error in shutil.which")
        tool_installed_evaluation = ToolInstalledEvaluation()

        results, errors = asyncio.run(
            tool_installed_evaluation.run(tools=["python"])
        )

        self.assertEqual(
            len(results),
            0,
            "There should be no results when an exception occurs in shutil.which.",
        )
        self.assertEqual(
            len(errors),
            1,
            "There should be one error when an exception occurs in shutil.which.",
        )

        actual_error = errors[0]

        self.assertIn(
            "An error occurred while checking tool 'python': Unexpected error in shutil.which",
            actual_error.message,
            "Expected error message for the exception was not returned.",
        )
        self.assertEqual(
            actual_error.type,
            "Exception",
            "Expected error type for the exception was not returned.",
        )
