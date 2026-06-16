import tempfile
from unittest import TestCase
from pathlib import Path

from bxagent.preparation.explore_models import create_explore_models_node


class TestExploreModels(TestCase):
    def setUp(self):
        self.explore_models = create_explore_models_node()

    def test_explore_models__models_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_model_path = Path(temp_dir) / "source_model.py"
            target_model_path = Path(temp_dir) / "target_model.py"

            source_model_implementation = "print('Hello from source model')"
            target_model_implementation = "print('Hello from target model')"

            source_model_path.write_text(source_model_implementation)
            target_model_path.write_text(target_model_implementation)

            result = self.explore_models(
                {
                    "source_model_path": source_model_path,
                    "target_model_path": target_model_path,
                }
            )

            self.assertEqual(
                result["source_model_implementation"],
                source_model_implementation,
                "Source model implementation should match the content of the source model file.",
            )
            self.assertEqual(
                result["target_model_implementation"],
                target_model_implementation,
                "Target model implementation should match the content of the target model file.",
            )

    def test_explore_models__no_models(self):
        # No paths at all
        with self.assertRaises(ValueError) as context:
            self.explore_models({})

        with tempfile.TemporaryDirectory() as temp_dir:
            source_model_path = Path(temp_dir) / "source_model.py"
            target_model_path = Path(temp_dir) / "target_model.py"

            # Source model path does not exist
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )

            source_model_path.touch()  # Create an empty source model file

            # Target model path does not exist
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )

    def test_explore_models__no_implementation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_model_path = Path(temp_dir) / "source_model.py"
            target_model_path = Path(temp_dir) / "target_model.py"

            source_model_path.touch()  # Create an empty source model file
            target_model_path.touch()  # Create an empty target model file

            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )

            source_model_path.write_text(
                "Some implementation"
            )  # Write content to source model file

            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )
