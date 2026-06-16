import asyncio

from unittest import TestCase
from unittest.mock import patch

from bxagent.validation.implementations.command_installed import (
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
            1,
            "There should be one result for the installed command.",
        )
        self.assertEqual(
            len(errors),
            1,
            "There should be one error for the non-existent command.",
        )
        self.assertIn(
            "Command 'python' is installed on the system.",
            [result.content for result in results],
            "Expected success message for 'python' was not returned.",
        )
        self.assertIn(
            "Command 'nonexistentcommand' is not installed on the system.",
            [error.message for error in errors],
            "Expected error message for 'nonexistentcommand' was not returned.",
        )
