import shutil
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from mdeagent.comprehension import TransformationPlanData
from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.preparation.prepare_workspace import (
    StructureFixStrategy,
    create_prepare_workspace_node,
)
from mdeagent.preparation.state import PreparationState
from mdeagent.util import copy_workspace, log_workspace_structure


class TestPrepareWorkspace(TestCase):
    def setUp(self):
        self.maxDiff = None
        
        def fix_structure_side_effect(state: PreparationState) -> PreparationState:
            """Mock fix strategy that creates the missing pom.xml."""
            workspace = state.get("workspace_path")
            artifact_id = state.get("artifact_id")
            group_id = state.get("group_id")
            if workspace and artifact_id and group_id:
                # Create parent pom.xml
                parent_pom_path = workspace / "pom.xml"
                parent_pom_path.parent.mkdir(parents=True, exist_ok=True)
                parent_pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>parent</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
</project>""")
                # Create child pom.xml
                child_pom_path = workspace / artifact_id / "pom.xml"
                child_pom_path.parent.mkdir(parents=True, exist_ok=True)
                child_pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
            return PreparationState()
        
        self.fix_strategy = Mock(spec=StructureFixStrategy)
        self.fix_strategy.fix_structure.side_effect = fix_structure_side_effect
        self.prepare_workspace_node = create_prepare_workspace_node(self.fix_strategy)
        self.fake_data = TransformationPlanData(
            iteration=0,
            source_model_package="de.example.mdeagent",
            target_model_package="de.example.mdeagent",
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties="",
            implementation_steps="",
        )
        self.template_path = Path.cwd() / "templates"
    
    def _mock_subprocess_run(self, args, **kwargs):
        """Helper to mock subprocess.run and create minimal Maven project structure."""
        import subprocess
        cwd = kwargs.get('cwd', Path.cwd())
        
        # Check if this is an archetype:generate call
        if 'archetype:generate' in args:
            # Extract artifactId from args
            artifact_id = None
            group_id = None
            for arg in args:
                if arg.startswith('-DartifactId='):
                    artifact_id = arg.split('=')[1]
                elif arg.startswith('-DgroupId='):
                    group_id = arg.split('=')[1]
            
            if artifact_id and group_id:
                # Create the child project structure
                child_path = Path(cwd) / artifact_id
                child_path.mkdir(parents=True, exist_ok=True)
                
                # Create pom.xml
                pom_path = child_path / "pom.xml"
                pom_path.write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>{group_id}</groupId>
    <artifactId>{artifact_id}</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
                
                # Create src directory structure
                src_path = child_path / "src" / "main" / "java" / group_id.replace('.', '/')
                src_path.mkdir(parents=True, exist_ok=True)
                
                # Create App.java
                app_java = src_path / "App.java"
                app_java.write_text(f"package {group_id};\npublic class App {{}}")
        
        return subprocess.CompletedProcess(args=args, returncode=0)

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,  # Will be set per test
    )
    def test_prepare_workspace__given_folder_does_not_exist(
        self,
        mock_run: Mock,
    ):
        """
        This test checks if the workspace is created successfully if the given workspace folder does not exist
        """
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = (
                Path(temp_dir) / "workspace"
            )  # This folder is not created yet
            input_state = PreparationState(
                required_commands=[],
                workspace_path=workspace_path,
                group_id="de.example",
                artifact_id="mdeagent",
                # package_path="de.example.mdeagent",
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
            self.assertIsInstance(
                output_state.get("transformation_class_path"),
                Path,
                "The output state should contain the transformation class path.",
            )
            self.assertEqual(
                output_state.get("maven_project_path"),
                workspace_path / "mdeagent",
            )

            # Check if subprocess.run was called for creating child project with archetype and for validating
            self.assertEqual(mock_run.call_count, 2)

            # Check indirect output
            self.assertTrue(
                (Path(workspace_path) / "mdeagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(workspace_path) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(workspace_path) / "mdeagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__given_folder_exists(
        self,
        mock_run: Mock,
    ):
        """
        This test checks if the workspace is created successfully if the given workspace folder exists
        """
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                group_id="de.example",
                artifact_id="mdeagent",
                # package_path="de.example.mdeagent",
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
            self.assertIsInstance(
                output_state.get("transformation_class_path"),
                Path,
                "The output state should contain the transformation class path.",
            )
            self.assertEqual(
                output_state.get("maven_project_path"),
                Path(temp_dir) / "mdeagent",
            )
            # Check if subprocess.run was called for creating child project with archetype and for validating
            self.assertEqual(mock_run.call_count, 2)

            # Check indirect output
            self.assertTrue(
                (Path(temp_dir) / "mdeagent" / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "pom.xml").exists(),
            )
            self.assertTrue(
                (Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )

    def test_prepare_workspace__content_exists_structure_incorrect(
        self
    ):
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
                artifact_id="mdeagent",
            )

            # Create the workspace manually
            # There is no "pom.xml"
            (Path(temp_dir) / "mdeagent").mkdir(parents=True)
            (Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md").touch()
            (
                Path(temp_dir)
                / "mdeagent"
                / "src"
                / "main"
                / "java"
                / "de"
                / "example"
                / "mdeagent"
            ).mkdir(parents=True)

            self.prepare_workspace_node(input_state)
            self.fix_strategy.fix_structure.assert_called_once_with(input_state)

    @patch(
        "subprocess.run",
        side_effect=lambda *args, **kwargs: None,
    )
    def test_prepare_workspace__transformation_plan_exists(
        self,
        mock_run: Mock,
    ):
        # Configure mock to create project structure
        mock_run.side_effect = self._mock_subprocess_run
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake transformation plan in the workspace
            tp_path = Path(temp_dir) / "mdeagent" / "TRANSFORMATION.md"
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
            # Parent pom.xml
            (Path(temp_dir) / "pom.xml").write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>de.example</groupId>
    <artifactId>parent</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
</project>""")
            
            # Child project directory with pom.xml
            child_pom_path = Path(temp_dir) / "mdeagent" / "pom.xml"
            child_pom_path.parent.mkdir(parents=True, exist_ok=True)
            child_pom_path.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>
    <groupId>de.example</groupId>
    <artifactId>mdeagent</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>""")
            
            # src directory
            (Path(temp_dir) / "mdeagent" / "src" / "main" / "java" / "de" / "example" / "mdeagent").mkdir(parents=True, exist_ok=True)

            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                # package_path="de.example.mdeagent",
                group_id="de.example",
                artifact_id="mdeagent",
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

            # Should call subprocess.run for validating the existing Maven project, but not for creating a new one
            self.assertEqual(mock_run.call_count, 1)

    def test_prepare_workspace__state_properties_missing(self):
        input_state = PreparationState(
            required_commands=[],
            workspace_path=None,  # Missing workspace path
            # package_path="de.example.mdeagent",
            group_id="de.example",
            artifact_id="mdeagent",
        )

        with self.assertRaises(ValueError):
            self.prepare_workspace_node(input_state)

        input_state = PreparationState(
            required_commands=[],
            workspace_path=Path("/some/path"),
            group_id=None,
            artifact_id="mdeagent",
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
                artifact_id="mdeagent",
            )

            try:
                output = create_prepare_workspace_node(
                    fix_strategy=Mock(spec=StructureFixStrategy)
                )(input_state)

                self.assertEqual(
                    output.get("maven_project_path"),
                    Path(temp_dir) / "mdeagent",
                    "The output state should contain the maven project path.",
                )
            except Exception as e:
                log_workspace_structure(Path(temp_dir))
                copy_workspace(
                    Path(temp_dir),
                    Path.cwd() / ".mdeagent-workspace/prepare_workspace_test",
                )
                self.fail(
                    f"prepare_workspace_node raised an exception unexpectedly: {e}"
                )

            # Check for the correct structure
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                ).exists(),
                "The 'src/main/java/de/example/mdeagent' folder should be created in the workspace.",
            )

            # Check if the bxtool Java file is created
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                    / "MDEAgentTransformationBxToolAdapter.java"
                ).exists(),
                "The bxtool Java file should be created in the package path.",
            )

            # Check if the AgentTransformationForEMF.java file is created
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "mdeagent"
                    / "AgentTransformationForEMF.java"
                ).exists(),
                "The AgentTransformationForEMF.java file should be created in the package path.",
            )

            # Check if the App.java file is deleted
            self.assertFalse(
                (
                    Path(temp_dir)
                    / "mdeagent"
                    / "src"
                    / "main"
                    / "java"
                    / "de"
                    / "example"
                    / "App.java"
                ).exists(),
                "The App.java file should be deleted.",
            )
