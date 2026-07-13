import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from bxagent.comprehension import TransformationPlanData
from bxagent.comprehension.plan import TransformationPlan
from bxagent.preparation.prepare_workspace import (
    create_prepare_workspace_node,
    StructureFixStrategy,
)
from bxagent.preparation.state import PreparationState
from bxagent.util import copy_workspace, log_workspace_structure


class TestPrepareWorkspace(TestCase):
    def setUp(self):
        self.maxDiff = None
        self.fix_strategy = Mock(spec=StructureFixStrategy)
        self.fix_strategy.fix_structure.return_value = PreparationState()
        self.prepare_workspace_node = create_prepare_workspace_node(self.fix_strategy)
        self.fake_data = TransformationPlanData(
            iteration=0,
            source_model_package="de.example.bxagent",
            target_model_package="de.example.bxagent",
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties="",
            implementation_steps="",
        )
        self.template_path = Path.cwd() / "templates"

    @patch(
            "bxagent.preparation.pom.install_dependencies",
            return_value=None,
    )
    @patch(
            "bxagent.preparation.pom.add_dependencies_to_pom",
            return_value=None,
    )
    @patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["mvn", "archetype:generate"], returncode=0
        ),
    )
    def test_prepare_workspace__given_folder_does_not_exist(self, mock_run: Mock, mock_add_dependencies: Mock, mock_install_dependencies: Mock):
        """
        This test checks if the workspace is created successfully if the given workspace folder does not exist
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = (
                Path(temp_dir) / "workspace"
            )  # This folder is not created yet
            input_state = PreparationState(
                required_commands=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="bxagent",
                # package_path="de.example.bxagent",
            )

            output_state: PreparationState = self.prepare_workspace_node(input_state)

            # Check direct output
            self.assertIsInstance(
                output_state.get("transformation_plan"),
                TransformationPlan,
                "The output state should contain a transformation plan.",
            )
            self.assertIsInstance(
                output_state.get("bxtool_path"),
                Path,
                "The output state should contain the bxtool path.",
            )

            # Check if maven was called to create the project structure
            mock_run.assert_called_once()
            mock_add_dependencies.assert_called_once() 
            mock_install_dependencies.assert_called_once()

            # Check indirect output
            self.assertTrue(
                (Path(workspace_path) / "bxagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(workspace_path) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(workspace_path) / "bxagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    @patch(
            "bxagent.preparation.pom.install_dependencies",
            return_value=None,
    )
    @patch(
            "bxagent.preparation.pom.add_dependencies_to_pom",
            return_value=None,
    )
    @patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["mvn", "archetype:generate"], returncode=0
        ),
    )
    def test_prepare_workspace__given_folder_exists(self, mock_run: Mock, mock_add_dependencies: Mock, mock_install_dependencies: Mock):
        """
        This test checks if the workspace is created successfully if the given workspace folder exists
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="bxagent",
                # package_path="de.example.bxagent",
            )

            output_state: PreparationState = self.prepare_workspace_node(input_state)

            # Check direct output
            self.assertIsInstance(
                output_state.get("transformation_plan"),
                TransformationPlan,
                "The output state should contain a transformation plan.",
            )
            self.assertIsInstance(
                output_state.get("bxtool_path"),
                Path,
                "The output state should contain the bxtool path.",
            )

            # Check if maven was called to create the project structure
            mock_run.assert_called_once()
            mock_add_dependencies.assert_called_once()
            mock_install_dependencies.assert_called_once()

            # Check indirect output
            self.assertTrue(
                (Path(temp_dir) / "bxagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(temp_dir) / "bxagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    @patch(
            "bxagent.preparation.pom.install_dependencies",
            return_value=None,
    )
    @patch(
            "bxagent.preparation.pom.add_dependencies_to_pom",
            return_value=None,
    )
    def test_prepare_workspace__content_exists_structure_incorrect(self, mock_add_dependencies: Mock, mock_install_dependencies: Mock):
        """
        This test checks if the StructureFixStrategy is invoked if the workspace folder exists but the structure is incorrect.
        Some strategies would be:
        - Delete the existing content and create the structure again
        - Move the existing content to a backup folder and create the structure again
        - Merge the existing content with the new structure (if possible)
        - Abort the operation and ask the user to fix the structure manually
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="bxagent",
            )

            # Create the workspace manually
            # There is no "pom.xml"
            (Path(temp_dir) / "bxagent").mkdir(parents=True)
            (Path(temp_dir) / "bxagent" / "TRANSFORMATION.md").touch()
            (
                Path(temp_dir)
                / "bxagent"
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "bxagent"
            ).mkdir(parents=True)

            self.prepare_workspace_node(input_state)
            self.fix_strategy.fix_structure.assert_called_once_with(input_state)

            mock_add_dependencies.assert_called_once()
            mock_install_dependencies.assert_called_once()

    @patch(
            "bxagent.preparation.pom.install_dependencies",
            return_value=None,
    )
    @patch(
            "bxagent.preparation.pom.add_dependencies_to_pom",
            return_value=None,
    )
    @patch(
        "subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=["mvn", "archetype:generate"], returncode=0
        ),
    )
    def test_prepare_workspace__transformation_plan_exists(self, mock_run: Mock, mock_add_dependencies: Mock, mock_install_dependencies: Mock):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake transformation plan in the workspace
            tp_path = Path(temp_dir) / "bxagent" / "TRANSFORMATION.md"
            tp_path.parent.mkdir(parents=True, exist_ok=True)
            tp_path.touch()
            tp = TransformationPlan.from_dict(
                {
                    "data": self.fake_data,
                    "parser": {
                        "type": "FileTransformationPlanParser",
                        "args": {"file_path": str(tp_path)},
                    },
                    "template": self.template_path,
                }
            )
            tp.update_iteration(1)

            # Create the rest of the necessary structure for the workspace
            (Path(temp_dir) / "pom.xml").touch()

            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                # package_path="de.example.bxagent",
                group_id="de.example",
                artifact_id="bxagent",
            )

            output_state = self.prepare_workspace_node(input_state)
            actual_tp: TransformationPlan = output_state.get("transformation_plan")

            self.assertIsNotNone(
                actual_tp,
                "The output state should contain a transformation plan.",
            )

            self.assertEqual(
                actual_tp.to_dict(),
                tp.to_dict(),
                "The transformation plan in the output state should match the existing transformation plan.",
            )

            mock_run.assert_called_once()
            mock_add_dependencies.assert_called_once()
            mock_install_dependencies.assert_called_once()

    def test_prepare_workspace__state_properties_missing(self):
        input_state = PreparationState(
            required_commands=[],
            workspace_path=None,  # Missing workspace path
            # package_path="de.example.bxagent",
            group_id="de.example",
            artifact_id="bxagent",
        )

        with self.assertRaises(ValueError):
            self.prepare_workspace_node(input_state)

        input_state = PreparationState(
            required_commands=[],
            workspace_path=Path("/some/path"),
            group_id=None,
            artifact_id="bxagent",
        )

        with self.assertRaises(ValueError):
            self.prepare_workspace_node(input_state)

        input_state = PreparationState(
            required_commands=[],
            workspace_path=Path("/some/path"),
            group_id="de.example",
            artifact_id=None,
        )

        with self.assertRaises(ValueError):
            self.prepare_workspace_node(input_state)


class TestMavenIntegration(TestCase):
    def setUp(self):
        if not shutil.which("mvn"):
            self.skipTest("Maven is not installed. Skipping Maven integration tests.")

    def test_prepare_workspace__maven_project_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=["mvn"],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="bxagent",
            )

            try: 
                create_prepare_workspace_node(fix_strategy=Mock(spec=StructureFixStrategy))(
                    input_state
                )
            except Exception as e:
                log_workspace_structure(Path(temp_dir))
                copy_workspace(Path(temp_dir), Path.cwd() / ".bx-agent-workspace/prepare_workspace_test")
                self.fail(f"prepare_workspace_node raised an exception unexpectedly: {e}")

            # Check for the correct structure
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "bxagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "bxagent"
                ).exists(),
                "The 'src/main/java/de/example/bxagent' folder should be created in the workspace.",
            )

            # Check if the transformation Java file is created
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "bxagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "bxagent"
                    / "BxAgentJavaBxTool.java"
                ).exists(),
                "The transformation Java file should be created in the package path.",
            )

            # Check if the App.java file is deleted
            self.assertFalse(
                (
                    Path(temp_dir)
                    / "bxagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "bxagent"
                    / "App.java"
                ).exists(),
                "The App.java file should be deleted.",
            )
