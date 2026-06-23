import tempfile
from pathlib import Path
from unittest import TestCase

from bxagent.comprehension import TransformationPlanData
from bxagent.comprehension.plan import TransformationPlan
from bxagent.preparation.prepare_workspace import create_prepare_workspace_node
from bxagent.preparation.state import PreparationState


class TestPrepareWorkspace(TestCase):
    def setUp(self):
        self.maxDiff = None
        self.prepare_workspace_node = create_prepare_workspace_node()
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

    def test_prepare_workspace__create_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                package_path="de.example.bxagent",
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

            # Check indirect output
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
            self.assertTrue(
                (Path(temp_dir) / "src" / "de" / "example" / "bxagent" / "BxAgentJavaBxTool.java").exists(),
                "The transformation Java file should be created in the package path.",
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
            self.assertTrue(
                (Path(temp_dir) / "src" / "de" / "example" / "bxagent" / "BxAgentJavaBxTool.java").exists(),
                "The transformation Java file should be created in the package path.",
            )

    def test_prepare_workspace__transformation_plan_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            transformation_plan_data = self.fake_data

            tp_path = Path(temp_dir) / "TRANSFORMATION.md"
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

            input_state = PreparationState(
                required_commands=[],
                workspace_path=Path(temp_dir),
                package_path="de.example.bxagent",
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
