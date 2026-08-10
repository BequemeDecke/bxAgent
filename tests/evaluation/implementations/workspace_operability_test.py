import asyncio
import tempfile

from pathlib import Path
from unittest import TestCase
from mdagent.evaluation.implementations.workspace_operability import (
    WorkspaceOperabilityEvaluation,
)


class TestWorkspaceOperability(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(WorkspaceOperabilityEvaluation, "setup"),
            "WorkspaceOperabilityEvaluation should have a 'setup' method.",
        )

        workspace_operability_evaluation = WorkspaceOperabilityEvaluation()

        self.assertIsNone(
            asyncio.run(workspace_operability_evaluation.setup()),
            "WorkspaceOperabilityEvaluation's 'setup' method should return None.",
        )

    def test_workspace_operability__no_workspace_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "missing-workspace"

            workspace_operability_evaluation = WorkspaceOperabilityEvaluation()
            results, errors = asyncio.run(
                workspace_operability_evaluation.run(
                    workspace_path=workspace_path,
                    package_path="de.example.mdagent",
                )
            )

            self.assertEqual(
                len(results),
                4,
                "There should be four failing results when the workspace folder is missing.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the workspace folder is missing.",
            )
            self.assertIn(
                f"Workspace path '{workspace_path}' does not exist or is not a directory.",
                [result.content for result in results],
                "Expected workspace path error was not returned.",
            )
            self.assertIn(
                "Required file 'TRANSFORMATION.md' is missing in the workspace.",
                [result.content for result in results],
                "Expected TRANSFORMATION.md error was not returned.",
            )
            self.assertIn(
                "Required folder 'src' is missing in the workspace.",
                [result.content for result in results],
                "Expected src folder error was not returned.",
            )
            self.assertIn(
                f"Package path 'de.example.mdagent' is invalid. Missing directory: '{workspace_path / 'src' / 'de'}'.",
                [result.content for result in results],
                "Expected package path error was not returned.",
            )

    def test_workspace_operability__invalid_transformation_md(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            package_path = workspace_path / "src" / "de" / "example"
            package_path.mkdir(parents=True)

            workspace_operability_evaluation = WorkspaceOperabilityEvaluation()
            results, errors = asyncio.run(
                workspace_operability_evaluation.run(
                    workspace_path=workspace_path,
                    package_path="de.example",
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failed result when TRANSFORMATION.md is missing.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when TRANSFORMATION.md is missing.",
            )
            self.assertEqual(
                results[0].content,
                "Required file 'TRANSFORMATION.md' is missing in the workspace.",
                "Expected TRANSFORMATION.md error message does not match.",
            )

    def test_workspace_operability__invalid_package_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir)
            (workspace_path / "TRANSFORMATION.md").write_text("# transformation")
            package_path = workspace_path / "src" / "de" / "example"
            package_path.mkdir(parents=True)

            workspace_operability_evaluation = WorkspaceOperabilityEvaluation()
            results, errors = asyncio.run(
                workspace_operability_evaluation.run(
                    workspace_path=workspace_path,
                    package_path="de.example.mdagent",
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failed result when the package path is invalid.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the package path is invalid.",
            )
            self.assertEqual(
                results[0].content,
                f"Package path 'de.example.mdagent' is invalid. Missing directory: '{workspace_path / 'src' / 'de' / 'example' / 'mdagent'}'.",
                "Expected package path error message does not match.",
            )
