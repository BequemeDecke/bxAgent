import tempfile
from unittest import TestCase
from pathlib import Path

from bxagent.preparation.explore_models import (
    create_explore_models_node,
    read_generated_emf_implementations,
)


def create_test_model_package(temp_dir: Path, package_name: str):
    package_path = temp_dir / package_name
    package_path.mkdir()

    source_file = package_path / f"{package_name}.java"
    source_register_file = package_path / f"{package_name}Register.java"
    source_package_file = package_path / f"{package_name}Package.java"
    source_factory_file = package_path / f"{package_name}Factory.java"

    source_file.write_text(f"public interface {package_name} {{ }}")
    source_register_file.write_text(f"public interface {package_name}Register {{ }}")
    source_package_file.write_text(f"public interface {package_name}Package {{ }}")
    source_factory_file.write_text(f"public interface {package_name}Factory {{ }}")
    return (
        package_path,
        source_file,
        source_register_file,
        source_package_file,
        source_factory_file,
    )


class TestExploreModels(TestCase):
    def setUp(self):
        self.explore_models = create_explore_models_node()

    def test_explore_models__models_exist(self):
        """
        Test, which checks if the explore_models function correctly follows the path and reads the content of the implementation of the model.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                source_model_path,
                source_file,
                source_register_file,
                source_package_file,
                source_factory_file,
            ) = create_test_model_package(Path(temp_dir), "Source")
            (
                target_model_path,
                target_file,
                target_register_file,
                target_package_file,
                target_factory_file,
            ) = create_test_model_package(Path(temp_dir), "Target")

            result = self.explore_models(
                {
                    "source_model_path": source_model_path,
                    "target_model_path": target_model_path,
                }
            )

            # Check if the source model implementation is read correctly
            self.assertIn(
                source_file.read_text(),
                result["source_model_implementation"],
                "Source model implementation should match the content of the source model file.",
            )
            self.assertIn(
                source_register_file.read_text(),
                result["source_model_implementation"],
                "Source model implementation should match the content of the source register file.",
            )
            self.assertIn(
                source_package_file.read_text(),
                result["source_model_implementation"],
                "Source model implementation should match the content of the source package file.",
            )
            self.assertIn(
                source_factory_file.read_text(),
                result["source_model_implementation"],
                "Source model implementation should match the content of the source factory file.",
            )
            # Check if the target model implementation is read correctly
            self.assertIn(
                target_file.read_text(),
                result["target_model_implementation"],
                "Target model implementation should match the content of the target model file.",
            )
            self.assertIn(
                target_register_file.read_text(),
                result["target_model_implementation"],
                "Target model implementation should match the content of the target register file.",
            )
            self.assertIn(
                target_package_file.read_text(),
                result["target_model_implementation"],
                "Target model implementation should match the content of the target package file.",
            )
            self.assertIn(
                target_factory_file.read_text(),
                result["target_model_implementation"],
                "Target model implementation should match the content of the target factory file.",
            )

    def test_explore_models__no_models(self):
        # No paths at all
        with self.assertRaises(ValueError) as context:
            self.explore_models({})

        with tempfile.TemporaryDirectory() as temp_dir:
            source_model_path = Path(temp_dir) / "Source"
            target_model_path = Path(temp_dir) / "Target"

            # Source model path does not exist
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )

            source_model_path.mkdir()  # Create an empty source model folder
            target_model_path.touch()  # Create an empty target model file

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
            source_model_path = Path(temp_dir) / "Source"
            target_model_path = Path(temp_dir) / "Target"

            source_model_path.mkdir()  # Create an empty source model folder
            target_model_path.mkdir()  # Create an empty target model folder

            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "source_model_path": source_model_path,
                        "target_model_path": target_model_path,
                    }
                )


class TestReadGeneratedEMFImplementations(TestCase):
    def test_read_generated_emf_implementations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "gen" / "Source"
            package_path.mkdir(parents=True, exist_ok=True)

            # Create some dummy Java files in the package
            source_file = package_path / "Source.java"
            source_register_file = package_path / "SourceRegister.java"
            source_package_file = package_path / "SourcePackage.java"
            source_factory_file = package_path / "SourceFactory.java"

            source_file.write_text("public interface Source { }")
            source_register_file.write_text("public interface SourceRegister { }")
            source_package_file.write_text("public interface SourcePackage { }")
            source_factory_file.write_text("public interface SourceFactory { }")

            # Call the function to read the generated EMF implementations
            result = read_generated_emf_implementations(package_path)

            # Check if the result contains the expected content
            self.assertEqual(result[source_file], "public interface Source { }")
            self.assertEqual(
                result[source_register_file], "public interface SourceRegister { }"
            )
            self.assertEqual(
                result[source_package_file], "public interface SourcePackage { }"
            )
            self.assertEqual(
                result[source_factory_file], "public interface SourceFactory { }"
            )

    def test_read_generated_emf_implementations_empty_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            package_path = Path(temp_dir) / "gen" / "EmptyPackage"
            package_path.mkdir(parents=True, exist_ok=True)

            # Call the function to read the generated EMF implementations
            result = read_generated_emf_implementations(package_path)

            # Check if the result is an empty dictionary
            self.assertEqual(result, {})
