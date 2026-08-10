import asyncio
from unittest import TestCase
from unittest.mock import patch

from bxagent.evaluation.implementations.command_installed import (
    CommandInstalledValidation,
)


class CommandInstalled(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(CommandInstalledValidation, "setup"),
            "CommandInstalledValidation should have a 'setup' method.",
        )

        command_installed_validation = CommandInstalledValidation()

        self.assertIsNone(
            asyncio.run(command_installed_validation.setup()),
            "CommandInstalledValidation's 'setup' method should return None.",
        )

    @patch("shutil.which")
    def test_run__check_commands(self, mock_which):
        mock_which.side_effect = lambda command: (
            "/usr/bin/python" if command == "python" else None
        )
        command_installed_validation = CommandInstalledValidation()

        results, errors = asyncio.run(
            command_installed_validation.run(commands=["python", "nonexistentcommand"])
        )

        self.assertEqual(
            len(results),
            2,
            "There should be two results for the validation.",
        )
        self.assertEqual(
            len(errors),
            0,
            "There should be no errors occurred during validation.",
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
                if "nonexistentcommand" in result.content
            )
        )

    @patch("shutil.which")
    def test_run__exception_in_which(self, mock_which):
        mock_which.side_effect = Exception("Unexpected error in shutil.which")
        command_installed_validation = CommandInstalledValidation()

        results, errors = asyncio.run(
            command_installed_validation.run(commands=["python"])
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
            "An error occurred while checking command 'python': Unexpected error in shutil.which",
            actual_error.message,
            "Expected error message for the exception was not returned.",
        )
        self.assertEqual(
            actual_error.type,
            "Exception",
            "Expected error type for the exception was not returned.",
        )
