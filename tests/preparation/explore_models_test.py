import shutil
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from mdeagent.preparation.explore_models import (
    copy_model_to_workspace,
    create_explore_models_node,
    read_generated_emf_implementations,
)
from mdeagent.preparation.pom import Pom
from mdeagent.preparation.state import ModelImplementation, PreparationState


def create_test_model_package(temp_dir: Path, package_name: str):
    """
    Creates a test model package with Java files in the gen folder and a pom.xml.
    EMF-generated implementations are located in the gen subfolder.
    
    Args:
        temp_dir: The temporary directory root.
        package_name: The name of the package (e.g., "Families", "Persons").
        
    Returns:
        Tuple of (package_path, source_file, register_file, package_file, factory_file)
    """
    package_path = temp_dir / package_name
    package_path.mkdir()
    
    # EMF-generated code goes into the gen subfolder
    gen_path = package_path / "gen"
    gen_path.mkdir()

    source_file = gen_path / f"{package_name}.java"
    source_register_file = gen_path / f"{package_name}Register.java"
    source_package_file = gen_path / f"{package_name}Package.java"
    source_factory_file = gen_path / f"{package_name}Factory.java"

    source_file.write_text(f"public interface {package_name} {{ }}")
    source_register_file.write_text(f"public interface {package_name}Register {{ }}")
    source_package_file.write_text(f"public interface {package_name}Package {{ }}")
    source_factory_file.write_text(f"public interface {package_name}Factory {{ }}")
    
    # Create a minimal pom.xml for the model package (in root, not in gen)
    pom_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example.models</groupId>
  <artifactId>{package_name}</artifactId>
  <version>1.0</version>
  <packaging>jar</packaging>
  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>
</project>'''
    (package_path / "pom.xml").write_text(pom_content)
    
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
        Module names are derived from the path stem (e.g., /path/Families → Families).
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workspace = temp_path / "workspace"
            workspace.mkdir()
            
            # Create parent pom.xml
            parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules>
  </modules>
  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>
</project>'''
            (workspace / "pom.xml").write_text(parent_pom_content)
            
            # Model names derived from path: /path/Families → "Families"
            (
                source_model_path,
                source_file,
                source_register_file,
                source_package_file,
                source_factory_file,
            ) = create_test_model_package(temp_path, "Families")
            (
                target_model_path,
                target_file,
                target_register_file,
                target_package_file,
                target_factory_file,
            ) = create_test_model_package(temp_path, "Persons")

            result = self.explore_models(
                PreparationState(
                    workspace_path=workspace,
                    group_id="com.example",
                    source_model=ModelImplementation(
                        name="Families",
                        path=source_model_path,
                    ),
                    target_model=ModelImplementation(
                        name="Persons",
                        path=target_model_path,
                    ),
                )
            )

            # Check if the source model was copied to workspace (using derived name)
            source_model_workspace_path = workspace / "Families"
            self.assertTrue(source_model_workspace_path.exists(), "Source model should be copied to workspace/Families")
            self.assertTrue((source_model_workspace_path / "pom.xml").exists(), "Source model should have a pom.xml")
            
            # Check if the target model was copied to workspace (using derived name)
            target_model_workspace_path = workspace / "Persons"
            self.assertTrue(target_model_workspace_path.exists(), "Target model should be copied to workspace/Persons")
            self.assertTrue((target_model_workspace_path / "pom.xml").exists(), "Target model should have a pom.xml")
            
            # Check if modules are registered in parent pom.xml using derived names
            parent_pom = (workspace / "pom.xml").read_text()
            self.assertIn("<module>Families</module>", parent_pom, "Families should be registered as module")
            self.assertIn("<module>Persons</module>", parent_pom, "Persons should be registered as module")
            
            # Check if the source model implementation is read correctly
            self.assertIn(
                source_file.read_text(),
                result["source_model"]["implementation"],
                "Source model implementation should match the content of the source model file.",
            )
            self.assertIn(
                source_register_file.read_text(),
                result["source_model"]["implementation"],
                "Source model implementation should match the content of the source register file.",
            )
            self.assertIn(
                source_package_file.read_text(),
                result["source_model"]["implementation"],
                "Source model implementation should match the content of the source package file.",
            )
            self.assertIn(
                source_factory_file.read_text(),
                result["source_model"]["implementation"],
                "Source model implementation should match the content of the source factory file.",
            )
            # Check if the target model implementation is read correctly
            self.assertIn(
                target_file.read_text(),
                result["target_model"]["implementation"],
                "Target model implementation should match the content of the target model file.",
            )
            self.assertIn(
                target_register_file.read_text(),
                result["target_model"]["implementation"],
                "Target model implementation should match the content of the target register file.",
            )
            self.assertIn(
                target_package_file.read_text(),
                result["target_model"]["implementation"],
                "Target model implementation should match the content of the target package file.",
            )
            self.assertIn(
                target_factory_file.read_text(),
                result["target_model"]["implementation"],
                "Target model implementation should match the content of the target factory file.",
            )

    def test_explore_models__no_models(self):
        # No workspace_path at all
        with self.assertRaises(ValueError) as context:
            self.explore_models({})

        # Missing group_id
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "pom.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
</project>''')
            
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "workspace_path": workspace,
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workspace = temp_path / "workspace"
            workspace.mkdir()
            (workspace / "pom.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
</project>''')
            
            source_model_path = temp_path / "Source"
            target_model_path = temp_path / "Target"

            # Source model path does not exist
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "workspace_path": workspace,
                        "group_id": "com.example",
                        "source_model": ModelImplementation(
                            name="Source",
                            path=source_model_path,
                        ),
                        "target_model": ModelImplementation(
                            name="Target",
                            path=target_model_path,
                        ),
                    }
                )

            source_model_path.mkdir()  # Create an empty source model folder
            target_model_path.touch()  # Create an empty target model file

            # Target model path is not a directory
            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "workspace_path": workspace,
                        "group_id": "com.example",
                        "source_model": ModelImplementation(
                            name="Source",
                            path=source_model_path,
                        ),
                        "target_model": ModelImplementation(
                            name="Target",
                            path=target_model_path,
                        ),
                    }
                )

    def test_explore_models__no_implementation(self):
        """Test that explore_models raises an error when models have no Java files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            workspace = temp_path / "workspace"
            workspace.mkdir()
            
            # Create parent pom.xml
            parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
            (workspace / "pom.xml").write_text(parent_pom_content)
            
            # Create model directories with pom.xml but NO Java files
            source_model_path = temp_path / "EmptySource"
            source_model_path.mkdir()
            (source_model_path / "pom.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example.models</groupId>
  <artifactId>EmptySource</artifactId>
  <version>1.0</version>
</project>''')
            
            target_model_path = temp_path / "EmptyTarget"
            target_model_path.mkdir()
            (target_model_path / "pom.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example.models</groupId>
  <artifactId>EmptyTarget</artifactId>
  <version>1.0</version>
</project>''')

            with self.assertRaises(ValueError) as context:
                self.explore_models(
                    {
                        "workspace_path": workspace,
                        "group_id": "com.example",
                        "source_model": ModelImplementation(
                            name="EmptySource",
                            path=source_model_path,
                        ),
                        "target_model": ModelImplementation(
                            name="EmptyTarget",
                            path=target_model_path,
                        ),
                    }
                )


class TestCopyModelToWorkspace(TestCase):
    """Test cases for the copy_model_to_workspace helper function."""

    def setUp(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_copy_model_to_workspace_copies_files(self):
        """Test that copy_model_to_workspace copies all files from model to workspace."""
        # Model name derived from path: /path/Families → "Families"
        model_path, src_file, reg_file, pkg_file, fac_file = create_test_model_package(
            self.temp_path, "Families"
        )
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        # Create parent pom.xml
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        # Call the function (module_name derived from model_path.stem)
        project = copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path,
            group_id="com.example",
        )
        
        # Check if the model was copied (using derived name)
        copied_path = workspace / "Families"
        gen_path = copied_path / "gen"
        self.assertTrue(copied_path.exists())
        self.assertTrue(gen_path.exists())
        self.assertTrue((gen_path / "Families.java").exists())
        self.assertTrue((gen_path / "FamiliesRegister.java").exists())
        self.assertTrue((gen_path / "FamiliesPackage.java").exists())
        self.assertTrue((gen_path / "FamiliesFactory.java").exists())
        
        # Verify file contents match (files are in gen subfolder)
        self.assertEqual(
            (gen_path / "Families.java").read_text(),
            src_file.read_text(),
        )

    def test_copy_model_to_workspace_uses_existing_pom(self):
        """Test that copy_model_to_workspace uses the existing pom.xml from the model."""
        # Model name derived from path: /path/Persons → "Persons"
        model_path, _, _, _, _ = create_test_model_package(self.temp_path, "Persons")
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        project = copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path,
            group_id="com.example",
        )
        
        pom_path = workspace / "Persons" / "pom.xml"
        self.assertTrue(pom_path.exists())
        
        # Check that the original pom.xml is preserved (with EMF dependencies added)
        pom_content = pom_path.read_text()
        self.assertIn("<groupId>com.example.models</groupId>", pom_content)  # Original groupId
        self.assertIn("<artifactId>Persons</artifactId>", pom_content)  # Derived from path

    def test_copy_model_to_workspace_adds_emf_dependencies(self):
        """Test that copy_model_to_workspace adds EMF dependencies to the pom.xml."""
        model_path, _, _, _, _ = create_test_model_package(self.temp_path, "EcoreModel")
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        project = copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path,
            group_id="com.example",
        )
        
        # Save pom to disk and reload to check dependencies
        project.pom.save()
        
        pom_content = (workspace / "EcoreModel" / "pom.xml").read_text()
        self.assertIn("org.eclipse.emf", pom_content)
        self.assertIn("org.eclipse.emf.ecore", pom_content)
        self.assertIn("org.eclipse.emf.common", pom_content)
        self.assertIn("org.eclipse.emf.ecore.xmi", pom_content)

    def test_copy_model_to_workspace_returns_maven_project(self):
        """Test that copy_model_to_workspace returns a MavenProject instance."""
        from mdeagent.preparation.maven import MavenProject
        
        model_path, _, _, _, _ = create_test_model_package(self.temp_path, "PetriNet")
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        project = copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path,
            group_id="com.example",
        )
        
        self.assertIsInstance(project, MavenProject)
        self.assertEqual(project.workspace, workspace / "PetriNet")

    def test_copy_model_to_workspace_replaces_existing(self):
        """Test that copy_model_to_workspace replaces an existing module."""
        model_path_v1, _, _, _, _ = create_test_model_package(self.temp_path, "GraphModel")
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        # First copy
        copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path_v1,
            group_id="com.example",
        )
        
        # Create modified model with same name
        model_path_v2 = self.temp_path / "GraphModel"
        shutil.rmtree(model_path_v2)  # Remove old one first
        model_path_v2.mkdir()
        (model_path_v2 / "NewFile.java").write_text("public class NewFile {}")
        # Add pom.xml for valid Maven project
        (model_path_v2 / "pom.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example.models</groupId>
  <artifactId>GraphModel</artifactId>
  <version>1.0</version>
</project>''')
        
        # Second copy (should replace)
        copy_model_to_workspace(
            workspace=workspace,
            model_path=model_path_v2,
            group_id="com.example",
        )
        
        # Old files should be gone, new file should exist
        self.assertFalse((workspace / "GraphModel" / "GraphModel.java").exists())
        self.assertTrue((workspace / "GraphModel" / "NewFile.java").exists())

    def test_copy_model_to_workspace_raises_error_without_pom(self):
        """Test that copy_model_to_workspace raises FileNotFoundError when pom.xml is missing."""
        # Create model directory without pom.xml
        model_path = self.temp_path / "NoPomModel"
        model_path.mkdir()
        (model_path / "SomeClass.java").write_text("public class SomeClass {}")
        
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        with self.assertRaises(FileNotFoundError) as context:
            copy_model_to_workspace(
                workspace=workspace,
                model_path=model_path,
                group_id="com.example",
            )
        
        self.assertIn("pom.xml", str(context.exception))
        self.assertIn(str(model_path), str(context.exception))


class TestExploreModelsValidation(TestCase):
    """Test cases for Maven project validation in explore_models."""

    def setUp(self):
        self.explore_models = create_explore_models_node()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_explore_models_validates_parent_project(self):
        """Test that explore_models validates the parent Maven project."""
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        # Create a valid parent pom.xml
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        # Module names derived from paths: /path/UML → "UML", /path/BPMN → "BPMN"
        source_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "UML")
        target_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "BPMN")
        
        # Should not raise any validation errors
        result = self.explore_models(
            PreparationState(
                workspace_path=workspace,
                group_id="com.example",
                source_model=ModelImplementation(name="UML", path=source_model_path),
                target_model=ModelImplementation(name="BPMN", path=target_model_path),
            )
        )
        
        # Verify that models were processed
        self.assertIn("implementation", result["source_model"])
        self.assertIn("implementation", result["target_model"])

    def test_explore_models_validates_model_projects(self):
        """Test that explore_models validates the model Maven projects."""
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        source_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "SysML")
        target_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "AADL")
        
        # Execute explore_models - should validate both model projects
        self.explore_models(
            PreparationState(
                workspace_path=workspace,
                group_id="com.example",
                source_model=ModelImplementation(name="SysML", path=source_model_path),
                target_model=ModelImplementation(name="AADL", path=target_model_path),
            )
        )
        
        # Verify that directories exist with derived names and pom.xml
        self.assertTrue((workspace / "SysML" / "pom.xml").exists())
        self.assertTrue((workspace / "AADL" / "pom.xml").exists())


class TestExploreModelsModuleRegistration(TestCase):
    """Test cases for module registration in explore_models."""

    def setUp(self):
        self.explore_models = create_explore_models_node()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_explore_models_registers_modules_with_derived_names(self):
        """Test that explore_models registers modules using names derived from path stem."""
        workspace = self.temp_path / "workspace"
        workspace.mkdir()
        
        # Create parent pom.xml without modules
        parent_pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>parent</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>
  <modules></modules>
</project>'''
        (workspace / "pom.xml").write_text(parent_pom_content)
        
        # Module names derived from paths: /path/Families → "Families", /path/Persons → "Persons"
        source_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "Families")
        target_model_path, _, _, _, _ = create_test_model_package(self.temp_path, "Persons")
        
        self.explore_models(
            PreparationState(
                workspace_path=workspace,
                group_id="com.example",
                source_model=ModelImplementation(name="Families", path=source_model_path),
                target_model=ModelImplementation(name="Persons", path=target_model_path),
            )
        )
        
        # Reload parent pom and check modules
        parent_pom = Pom(workspace / "pom.xml")
        module_ids = [m.artifact_id for m in parent_pom.modules]
        
        self.assertIn("Families", module_ids, "Module name should be derived from path stem")
        self.assertIn("Persons", module_ids, "Module name should be derived from path stem")
        
        # Verify directories exist with correct names
        self.assertTrue((workspace / "Families").exists())
        self.assertTrue((workspace / "Persons").exists())


class TestReadGeneratedEMFImplementations(TestCase):
    def test_read_generated_emf_implementations_reads_from_gen_folder(self):
        """Test that read_generated_emf_implementations reads Java files from the gen subfolder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Model package structure: /temp/Families/gen/*.java
            package_path = Path(temp_dir) / "Families"
            package_path.mkdir()
            gen_path = package_path / "gen"
            gen_path.mkdir()

            # Create Java files in the gen folder (EMF-generated code location)
            families_file = gen_path / "Families.java"
            families_register_file = gen_path / "FamiliesRegister.java"
            families_package_file = gen_path / "FamiliesPackage.java"
            families_factory_file = gen_path / "FamiliesFactory.java"

            families_file.write_text("public interface Families { }")
            families_register_file.write_text("public interface FamiliesRegister { }")
            families_package_file.write_text("public interface FamiliesPackage { }")
            families_factory_file.write_text("public interface FamiliesFactory { }")

            # Call the function with the package root path (it should look in gen/)
            result = read_generated_emf_implementations(package_path)

            # Check if files from gen folder are found
            self.assertEqual(len(result), 4)
            self.assertEqual(result[families_file], "public interface Families { }")
            self.assertEqual(
                result[families_register_file], "public interface FamiliesRegister { }"
            )
            self.assertEqual(
                result[families_package_file], "public interface FamiliesPackage { }"
            )
            self.assertEqual(
                result[families_factory_file], "public interface FamiliesFactory { }"
            )

    def test_read_generated_emf_implementations_empty_gen_folder(self):
        """Test that read_generated_emf_implementations returns empty dict for empty gen folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Model package with empty gen folder
            package_path = Path(temp_dir) / "EmptyModel"
            package_path.mkdir()
            gen_path = package_path / "gen"
            gen_path.mkdir()

            # Call the function - should return empty dict since gen has no Java files
            result = read_generated_emf_implementations(package_path)

            self.assertEqual(result, {})

    def test_read_generated_emf_implementations_fallback_without_gen(self):
        """Test that read_generated_emf_implementations falls back to package root if no gen folder exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Package without gen folder
            package_path = Path(temp_dir) / "NoGenPackage"
            package_path.mkdir()

            # Create Java file directly in package root
            java_file = package_path / "DirectFile.java"
            java_file.write_text("public class DirectFile {}")

            # Should fall back to reading from package root
            result = read_generated_emf_implementations(package_path)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[java_file], "public class DirectFile {}")
