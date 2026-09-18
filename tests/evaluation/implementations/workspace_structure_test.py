import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase

from mdeagent.comprehension import FileTransformationPlanParser
from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.implementations.workspace_structure import (
    AGENT_TRANSFORMATION_CLASS_NAME,
    SPOTLESS_PLUGIN_ARTIFACT_ID,
    WorkspaceStructureEvaluation,
)
from mdeagent.preparation.pom import Module, Plugin, Pom
from mdeagent.preparation.prepare_workspace import SPOTLESS_PLUGIN

# Path to the project's Jinja templates, resolved relative to this test file so
# that the tests do not depend on the current working directory.
TEMPLATES_PATH = Path(__file__).resolve().parents[3] / "templates"

GROUP_ID = "de.example"
ARTIFACT_ID = "mdagent"
PACKAGE_PATH = f"{GROUP_ID}.{ARTIFACT_ID}"


def _create_valid_workspace(
    workspace_path: Path,
    group_id: str = GROUP_ID,
    artifact_id: str = ARTIFACT_ID,
    package_path: str = PACKAGE_PATH,
) -> Path:
    """Create a fully valid workspace structure and return the module path.

    The created structure matches the one produced by the preparation node:
    ```
    workspace/pom.xml                                  (registers the module)
    workspace/[artifact_id]/pom.xml                    (registers the Spotless plugin)
    workspace/[artifact_id]/src/main/java/[pkg]/AgentTransformationForEMF.java
    workspace/[artifact_id]/TRANSFORMATION.md          (valid transformation plan)
    ```
    """
    # 1. Parent pom.xml with the module registered.
    Pom.new(workspace_path, group_id, workspace_path.name).add_module(
        Module(artifact_id)
    ).save()

    # 2. Module pom.xml with the Spotless plugin registered.
    module_path = workspace_path / artifact_id
    Pom.new(module_path, group_id, artifact_id).add_plugin(SPOTLESS_PLUGIN).save()

    # 3. Package path and AgentTransformationForEMF.java.
    package_dir = module_path / "src" / "main" / "java"
    for part in package_path.split("."):
        package_dir = package_dir / part
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / f"{AGENT_TRANSFORMATION_CLASS_NAME}.java").write_text(
        f"public class {AGENT_TRANSFORMATION_CLASS_NAME} {{ /* stub */ }}"
    )

    # 4. Valid TRANSFORMATION.md (created from the project template).
    transformation_md_path = module_path / "TRANSFORMATION.md"
    TransformationPlan.parse(
        parser=FileTransformationPlanParser(transformation_md_path),
        template_path=TEMPLATES_PATH,
    )

    return module_path


class TestWorkspaceStructure(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(WorkspaceStructureEvaluation, "setup"),
            "WorkspaceStructureEvaluation should have a 'setup' method.",
        )

        workspace_structure_evaluation = WorkspaceStructureEvaluation()

        self.assertIsNone(
            asyncio.run(workspace_structure_evaluation.setup()),
            "WorkspaceStructureEvaluation's 'setup' method should return None.",
        )

    def test_workspace_structure__valid_structure(self):
        """A correctly prepared workspace should produce no results and no errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            _create_valid_workspace(workspace_path)

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                0,
                "There should be no failing results for a valid workspace structure.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors for a valid workspace structure.",
            )

    def test_workspace_structure__no_workspace_folder(self):
        """If the workspace path does not exist, only the workspace path result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "missing-workspace"

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be a single failing result when the workspace folder is missing.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the workspace folder is missing.",
            )
            self.assertEqual(
                results[0].content,
                f"Workspace path '{workspace_path}' does not exist or is not a directory.",
                "Expected workspace path error was not returned.",
            )

    def test_workspace_structure__module_not_loadable(self):
        """If the module pom.xml is missing, MavenProject.load fails.
        
        This is expected behavior in early iterations when the Maven project
        hasn't been created yet. The evaluation should return a result with
        include_in_report=True so it's visible in the report.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            module_path = _create_valid_workspace(workspace_path)

            # Remove the module pom.xml so that MavenProject.load fails.
            (module_path / "pom.xml").unlink()

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failing result when the Maven project cannot be loaded.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the Maven project cannot be loaded.",
            )
            # Check that the error message indicates Maven project not found
            self.assertTrue(
                "Maven project" in results[0].content and (
                    results[0].content.startswith(f"Maven project not found at '{module_path}'") or
                    results[0].content.startswith(f"Failed to load Maven project at '{module_path}'")
                ),
                "Expected Maven project load error was not returned.",
            )
            # Verify this is marked for reporting (expected behavior, not a system error)
            self.assertTrue(
                results[0].metadata.get("include_in_report", False),
                "Maven project load failure should be included in report as expected behavior.",
            )

    def test_workspace_structure__module_not_registered_in_parent_pom(self):
        """If the module is not registered in the parent pom.xml, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            _create_valid_workspace(workspace_path)

            # Recreate the parent pom.xml without registering the module.
            Pom.new(workspace_path, GROUP_ID, workspace_path.name).save()

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failing result when the module is not registered.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the module is not registered.",
            )
            self.assertEqual(
                results[0].content,
                f"Module '{ARTIFACT_ID}' is not registered in '{workspace_path / 'pom.xml'}'.",
                "Expected module-not-registered error was not returned.",
            )

    def test_workspace_structure__spotless_plugin_missing(self):
        """If the Spotless plugin is not registered in the module pom.xml, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            module_path = _create_valid_workspace(workspace_path)

            # Recreate the module pom.xml without the Spotless plugin.
            Pom.new(module_path, GROUP_ID, ARTIFACT_ID).save()

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failing result when the Spotless plugin is missing.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when the Spotless plugin is missing.",
            )
            self.assertEqual(
                results[0].content,
                f"Spotless plugin '{SPOTLESS_PLUGIN_ARTIFACT_ID}' is not registered in '{module_path / 'pom.xml'}'.",
                "Expected Spotless plugin missing error was not returned.",
            )

    def test_workspace_structure__invalid_package_path(self):
        """If the package path is invalid, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            _create_valid_workspace(workspace_path)

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path="de.example.mdagent.nonexistent",
                )
            )

            missing_directory = (
                workspace_path
                / ARTIFACT_ID
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "mdagent"
                / "nonexistent"
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
                f"Package path 'de.example.mdagent.nonexistent' is invalid. Missing directory: '{missing_directory}'.",
                "Expected package path error message does not match.",
            )

    def test_workspace_structure__agent_transformation_missing(self):
        """If the AgentTransformationForEMF.java file is missing, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            module_path = _create_valid_workspace(workspace_path)

            # Remove the AgentTransformationForEMF.java file.
            package_dir = (
                module_path
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "mdagent"
            )
            (package_dir / f"{AGENT_TRANSFORMATION_CLASS_NAME}.java").unlink()

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failed result when AgentTransformationForEMF.java is missing.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when AgentTransformationForEMF.java is missing.",
            )
            self.assertEqual(
                results[0].content,
                f"Required file '{AGENT_TRANSFORMATION_CLASS_NAME}.java' is missing in '{package_dir}'.",
                "Expected AgentTransformationForEMF.java missing error was not returned.",
            )

    def test_workspace_structure__transformation_md_missing(self):
        """If the TRANSFORMATION.md file is missing, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            module_path = _create_valid_workspace(workspace_path)

            # Remove the TRANSFORMATION.md file.
            (module_path / "TRANSFORMATION.md").unlink()

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
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
            self.assertTrue(
                results[0].content.startswith(
                    f"Required file 'TRANSFORMATION.md' is missing or invalid at '{module_path / 'TRANSFORMATION.md'}'"
                ),
                "Expected TRANSFORMATION.md missing error was not returned.",
            )

    def test_workspace_structure__transformation_md_invalid(self):
        """If the TRANSFORMATION.md file is malformed, a result is returned."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir) / "workspace"
            module_path = _create_valid_workspace(workspace_path)

            # Overwrite the TRANSFORMATION.md with invalid content.
            (module_path / "TRANSFORMATION.md").write_text("not a valid plan")

            workspace_structure_evaluation = WorkspaceStructureEvaluation()
            results, errors = asyncio.run(
                workspace_structure_evaluation.run(
                    workspace_path=workspace_path,
                    artifact_id=ARTIFACT_ID,
                    package_path=PACKAGE_PATH,
                )
            )

            self.assertEqual(
                len(results),
                1,
                "There should be one failed result when TRANSFORMATION.md is invalid.",
            )
            self.assertEqual(
                len(errors),
                0,
                "There should be no errors when TRANSFORMATION.md is invalid.",
            )
            self.assertTrue(
                results[0].content.startswith(
                    f"Required file 'TRANSFORMATION.md' is missing or invalid at '{module_path / 'TRANSFORMATION.md'}'"
                ),
                "Expected TRANSFORMATION.md invalid error was not returned.",
            )
