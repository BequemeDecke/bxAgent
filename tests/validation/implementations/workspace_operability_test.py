import asyncio
import tempfile

from pathlib import Path
from unittest import TestCase
from bxagent.validation.implementations.workspace_operability import (
    WorkspaceOperabilityValidation,
)


class TestWorkspaceOperability(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(WorkspaceOperabilityValidation, "setup"),
            "WorkspaceOperabilityValidation should have a 'setup' method.",
        )

        workspace_operability_validation = WorkspaceOperabilityValidation()

        self.assertIsNone(
            asyncio.run(workspace_operability_validation.setup()),
            "WorkspaceOperabilityValidation's 'setup' method should return None.",
        )

    def test_workspace_operability__no_workspace_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "missing-workspace"

            workspace_operability_validation = WorkspaceOperabilityValidation()
            results, errors = asyncio.run(
                workspace_operability_validation.run(
                    workspace_path=workspace_path,
                    package_path="de.example.bxagent",
                )
            )

            self.assertEqual(
                len(results),
                0,
                "There should be no results when the workspace folder is missing.",
            )
            self.assertEqual(
                len(errors),
                4,
                "There should be four errors when the workspace folder is missing.",
            )
            self.assertIn(
                f"Workspace path '{workspace_path}' does not exist or is not a directory.",
                [error.message for error in errors],
                "Expected workspace path error was not returned.",
            )
            self.assertIn(
                "Required file 'TRANSFORMATION.md' is missing in the workspace.",
                [error.message for error in errors],
                "Expected TRANSFORMATION.md error was not returned.",
            )
            self.assertIn(
                "Required folder 'src' is missing in the workspace.",
                [error.message for error in errors],
                "Expected src folder error was not returned.",
            )
            self.assertIn(
                f"Package path 'de.example.bxagent' is invalid. Missing directory: '{workspace_path / 'src' / 'de'}'.",
                [error.message for error in errors],
                "Expected package path error was not returned.",
            )

    def test_workspace_operability__invalid_transformation_md(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            package_path = workspace_path / "src" / "de" / "example"
            package_path.mkdir(parents=True)

            workspace_operability_validation = WorkspaceOperabilityValidation()
            results, errors = asyncio.run(
                workspace_operability_validation.run(
                    workspace_path=workspace_path,
                    package_path="de.example",
                )
            )

            self.assertEqual(
                len(results),
                0,
                "There should be no results when TRANSFORMATION.md is missing.",
            )
            self.assertEqual(
                len(errors),
                1,
                "There should be one error when TRANSFORMATION.md is missing.",
            )
            self.assertEqual(
                errors[0].message,
                "Required file 'TRANSFORMATION.md' is missing in the workspace.",
                "Expected TRANSFORMATION.md error message does not match.",
            )

    def test_workspace_operability__invalid_package_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            (workspace_path / "TRANSFORMATION.md").write_text("# transformation")
            package_path = workspace_path / "src" / "de" / "example"
            package_path.mkdir(parents=True)

            workspace_operability_validation = WorkspaceOperabilityValidation()
            results, errors = asyncio.run(
                workspace_operability_validation.run(
                    workspace_path=workspace_path,
                    package_path="de.example.bxagent",
                )
            )

            self.assertEqual(
                len(results),
                0,
                "There should be no results when the package path is invalid.",
            )
            self.assertEqual(
                len(errors),
                1,
                "There should be one error when the package path is invalid.",
            )
            self.assertEqual(
                errors[0].message,
                f"Package path 'de.example.bxagent' is invalid. Missing directory: '{workspace_path / 'src' / 'de' / 'example' / 'bxagent'}'.",
                "Expected package path error message does not match.",
            )
