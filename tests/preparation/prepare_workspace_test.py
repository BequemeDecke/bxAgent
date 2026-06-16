import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch, Mock

from bxagent.preparation.prepare_workspace import create_prepare_workspace_node
from bxagent.comprehension import TransformationPlanData
from bxagent.preparation.state import PreparationState


class TestPrepareWorkspace(TestCase):
    def setUp(self):
        self.prepare_workspace_node = create_prepare_workspace_node()
        self.fake_data = TransformationPlanData(
            iteration=0,
            source_model_package="de.example.bxagent",
            target_model_package="de.example.bxagent",
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties=[],
            implementation_steps=[],
        )

    def test_prepare_workspace__create_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                package_path="de.example.bxagent",
            )

            self.prepare_workspace_node(input_state)

            self.assertTrue(
                (Path(temp_dir) / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "src" / "de" / "example" / "bxagent").exists(),
                "The package path should be created in the 'src' folder",
            )

    def test_prepare_workspace__workspace_already_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                package_path="de.example.bxagent",
            )
            # Create the workspace manually
            (Path(temp_dir) / "src").mkdir(parents=True)
            (Path(temp_dir) / "TRANSFORMATION.md").touch()
            (Path(temp_dir) / "src" / "de" / "example" / "bxagent").mkdir(parents=True)

            self.prepare_workspace_node(input_state)

            self.assertTrue(
                (Path(temp_dir) / "src").exists(),
                "The 'src' folder should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "TRANSFORMATION.md").exists(),
                "The 'TRANSFORMATION.md' file should be created in the workspace.",
            )
            self.assertTrue(
                (Path(temp_dir) / "src" / "de" / "example" / "bxagent").exists(),
                "The package path should be created in the 'src' folder",
            )
