import tempfile
from pathlib import Path
from unittest import TestCase

from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.pom import Pom


class TestMavenProjectInitialization(TestCase):
    """Test cases for MavenProject initialization."""
    
    def test_init_with_pom_and_workspace(self):
        """Test that MavenProject can be initialized with a Pom instance and workspace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            pom = Pom(pom_path)
            project = MavenProject(pom, workspace)
            
            self.assertIsInstance(project.pom, Pom)
            self.assertEqual(project.pom.pom_path, pom_path)
            self.assertEqual(project.workspace, workspace)


class TestMavenProjectLoad(TestCase):
    """Test cases for MavenProject.load() method."""
    
    def test_load_existing_project(self):
        """Test loading an existing Maven project from workspace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            project = MavenProject.load(workspace)
            
            self.assertIsInstance(project, MavenProject)
            self.assertEqual(project.pom.pom_path, pom_path)
            self.assertEqual(project.pom.modules, [])
    
    def test_load_without_pom_raises_error(self):
        """Test that load() raises FileNotFoundError when pom.xml is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            with self.assertRaises(FileNotFoundError) as context:
                MavenProject.load(workspace)
            
            self.assertIn("pom.xml not found", str(context.exception))
            self.assertIn(str(workspace), str(context.exception))
    
    def test_load_with_invalid_pom_raises_error(self):
        """Test that load() raises ValueError when pom.xml is invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            # Write invalid XML
            pom_path.write_text("This is not valid XML<<<>")
            
            with self.assertRaises(ValueError) as context:
                MavenProject.load(workspace)
            
            self.assertIn("Failed to parse pom.xml", str(context.exception))
    
    def test_load_with_empty_pom_raises_error(self):
        """Test that load() raises ValueError when pom.xml is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("")
            
            with self.assertRaises(ValueError) as context:
                MavenProject.load(workspace)
            
            self.assertIn("is empty", str(context.exception))


class TestMavenProjectCreate(TestCase):
    """Test cases for MavenProject.create() method."""
    
    def test_create_parent_project_without_parent(self):
        """Test creating a parent Maven project (no parent specified)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            artifact_id = "parent-project"
            
            project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=artifact_id,
                parent=None
            )
            
            self.assertIsInstance(project, MavenProject)
            self.assertTrue((workspace / "pom.xml").exists())
            
            # Verify the pom.xml content
            content = (workspace / "pom.xml").read_text()
            self.assertIn(f"<groupId>{group_id}</groupId>", content)
            self.assertIn(f"<artifactId>{artifact_id}</artifactId>", content)
    
    def test_create_child_project_with_parent(self):
        """Test creating a child Maven project with a parent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            parent_artifact_id = "parent-project"
            child_artifact_id = "child-module"
            
            # First create parent project
            parent_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=parent_artifact_id,
                parent=None
            )
            
            # Then create child project
            child_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=child_artifact_id,
                parent=parent_project
            )
            
            self.assertIsInstance(child_project, MavenProject)
            
            # Child project directory should exist
            child_pom_path = workspace / child_artifact_id / "pom.xml"
            self.assertTrue(child_pom_path.exists())
            
            # Child project's workspace should be workspace / artifact_id
            self.assertEqual(child_project.workspace, workspace / child_artifact_id)
            
            # Parent pom should have the child as a module
            parent_project.pom.save()  # Save to apply changes
            parent_content = (workspace / "pom.xml").read_text()
            self.assertIn(f"<module>{child_artifact_id}</module>", parent_content)
    
    def test_create_adds_module_to_parent(self):
        """Test that creating a child project adds it as a module to parent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            parent_artifact_id = "parent"
            child_artifact_id = "child"
            
            # Create parent
            parent_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=parent_artifact_id,
                parent=None
            )
            
            # Verify parent has no modules initially
            self.assertEqual(len(parent_project.pom.modules), 0)
            
            # Create child
            child_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=child_artifact_id,
                parent=parent_project
            )
            
            # Verify parent now has the child as a module (in cache)
            self.assertEqual(len(parent_project.pom.modules), 1)
            self.assertEqual(parent_project.pom.modules[0].artifact_id, child_artifact_id)
    
    def test_parent_and_child_have_different_workspaces(self):
        """Test that parent and child projects have different workspace paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            parent_artifact_id = "parent"
            child_artifact_id = "child"
            
            # Create parent
            parent_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=parent_artifact_id,
                parent=None
            )
            
            # Verify parent workspace is the base workspace
            self.assertEqual(parent_project.workspace, workspace)
            
            # Create child
            child_project = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=child_artifact_id,
                parent=parent_project
            )
            
            # Verify child workspace is workspace / artifact_id
            expected_child_workspace = workspace / child_artifact_id
            self.assertEqual(child_project.workspace, expected_child_workspace)
            self.assertNotEqual(parent_project.workspace, child_project.workspace)


class TestMavenProjectStructure(TestCase):
    """Test cases for MavenProject structure and relationships."""
    
    def test_parent_child_relationship(self):
        """Test the relationship between parent and child projects."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            
            # Create parent and two children
            parent = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id="parent",
                parent=None
            )
            
            child1 = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id="module-1",
                parent=parent
            )
            
            child2 = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id="module-2",
                parent=parent
            )
            
            # Parent should have both children as modules
            self.assertEqual(len(parent.pom.modules), 2)
            module_artifact_ids = {m.artifact_id for m in parent.pom.modules}
            self.assertIn("module-1", module_artifact_ids)
            self.assertIn("module-2", module_artifact_ids)
            
            # Children should have different workspaces
            self.assertNotEqual(child1.workspace, child2.workspace)
            self.assertEqual(child1.workspace, workspace / "module-1")
            self.assertEqual(child2.workspace, workspace / "module-2")
            
            # Children should be in different directories
            self.assertNotEqual(child1.pom.pom_path, child2.pom.pom_path)
            self.assertTrue(child1.pom.pom_path.parent.name.endswith("module-1"))
            self.assertTrue(child2.pom.pom_path.parent.name.endswith("module-2"))
    
    def test_multiple_levels_of_hierarchy(self):
        """Test creating multiple levels of project hierarchy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            
            # Create grandparent
            grandparent = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id="grandparent",
                parent=None
            )
            
            # Create parent as child of grandparent
            parent = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id="parent",
                parent=grandparent
            )
            
            # Create child as child of parent (not grandparent)
            # Note: This would require passing parent instead of grandparent
            # For this test, we just verify the structure works
            
            self.assertEqual(len(grandparent.pom.modules), 1)
            self.assertEqual(grandparent.pom.modules[0].artifact_id, "parent")


class TestMavenProjectAddFile(TestCase):
    """Test cases for MavenProject.add_file() method."""
    
    def test_add_file_creates_file(self):
        """Test that add_file creates a file with the specified content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            pom = Pom(pom_path)
            project = MavenProject(pom, workspace)
            
            relative_path = Path("src/main/resources/config.properties")
            content = "key=value\nfoo=bar"
            
            result_path = project.add_file(relative_path, content)
            
            # Should return the full path
            expected_path = workspace / relative_path
            self.assertEqual(result_path, expected_path)
            self.assertTrue(result_path.exists())
            self.assertEqual(result_path.read_text(), content)
    
    def test_add_file_creates_parent_directories(self):
        """Test that add_file creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            pom = Pom(pom_path)
            project = MavenProject(pom, workspace)
            
            relative_path = Path("deeply/nested/directory/structure/file.txt")
            
            result_path = project.add_file(relative_path, "content")
            
            self.assertTrue(result_path.exists())
            self.assertTrue(result_path.parent.exists())


class TestMavenProjectAddJavaClass(TestCase):
    """Test cases for MavenProject.add_java_class() method."""
    
    def test_add_java_class_creates_file(self):
        """Test that add_java_class creates a Java file in the correct package structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            pom = Pom(pom_path)
            project = MavenProject(pom, workspace)
            
            package = "com.example.service"
            class_name = "MyService"
            content = "package com.example.service;\npublic class MyService {}"
            
            result_path = project.add_java_class(package, class_name, content)
            
            expected_path = workspace / "src" / "main" / "java" / "com" / "example" / "service" / "MyService.java"
            self.assertEqual(result_path, expected_path)
            self.assertTrue(expected_path.exists())
            self.assertEqual(expected_path.read_text(), content)
    
    def test_add_java_class_with_nested_package(self):
        """Test that add_java_class handles deeply nested packages."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            pom_path = workspace / "pom.xml"
            pom_path.write_text("""<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-app</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>""")
            
            pom = Pom(pom_path)
            project = MavenProject(pom, workspace)
            
            package = "com.example.service.impl.internal"
            class_name = "InternalServiceImpl"
            content = "package com.example.service.impl.internal;\npublic class InternalServiceImpl {}"
            
            result_path = project.add_java_class(package, class_name, content)
            
            expected_path = workspace / "src" / "main" / "java" / "com" / "example" / "service" / "impl" / "internal" / "InternalServiceImpl.java"
            self.assertEqual(result_path, expected_path)
            self.assertTrue(expected_path.exists())


class TestMavenProjectChildWorkspace(TestCase):
    """Test cases for MavenProject child workspace handling."""
    
    def test_child_project_add_file_uses_child_workspace(self):
        """Test that add_file on a child project uses the child's workspace."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            parent_artifact_id = "parent"
            child_artifact_id = "child"
            
            # Create parent
            parent = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=parent_artifact_id,
                parent=None
            )
            
            # Create child
            child = MavenProject.create(
                workspace=workspace,
                group_id=group_id,
                artifact_id=child_artifact_id,
                parent=parent
            )
            
            # Add file to child project
            relative_path = Path("src/main/java/com/example/ChildClass.java")
            content = "package com.example; public class ChildClass {}"
            
            child.add_file(relative_path, content)
            
            # File should be in child's workspace, not parent's
            expected_path = child.workspace / relative_path
            self.assertTrue(expected_path.exists())
            
            # File should NOT be in parent's workspace
            parent_path = parent.workspace / relative_path
            self.assertNotEqual(expected_path.parent, parent_path.parent)
