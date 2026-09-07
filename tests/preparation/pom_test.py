import logging
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from mdeagent.preparation.pom import (
    Dependency,
    Module,
    Plugin,
    Pom,
)

logger = logging.getLogger(__name__)

INITIAL_POM = """<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>"""

INITIAL_POM_WITH_DEPENDENCIES = """<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <artifactId>workspace</artifactId>
    <groupId>de.example</groupId>
    <version>1.0</version>
  </parent>

  <groupId>de.example</groupId>
  <artifactId>mdagent</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>mdagent</name>
  <description>A simple mdagent.</description>
  
  <url>http://www.example.com</url>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>8</maven.compiler.source>
    <maven.compiler.target>8</maven.compiler.target>
  </properties>

  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>3.8.1</version>
    </dependency>
  </dependencies>

  <build>
    <pluginManagement>
      <plugins>
        <plugin>
          <artifactId>maven-clean-plugin</artifactId>
          <version>3.4.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-site-plugin</artifactId>
          <version>3.12.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-project-info-reports-plugin</artifactId>
          <version>3.6.1</version>
        </plugin>
      </plugins>
    </pluginManagement>
  </build>

  <reporting>
    <plugins>
      <plugin>
        <artifactId>maven-project-info-reports-plugin</artifactId>
      </plugin>
    </plugins>
  </reporting>
</project>"""


class TestPomInitialization(TestCase):
    """Test cases for Pom initialization and setup.
    
    This test case checks what happens when a pom.xml already exists.
    This test case checks what happens when a parent pom.xml will be created.
    """
    
    def test_initialize_pom(self):
        """
        Test that Pom can be initialized from an existing pom.xml file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            existing_pom_path = Path(temp_dir, "pom.xml")
            existing_pom_path.write_text(INITIAL_POM_WITH_DEPENDENCIES)
            
            pom = Pom(existing_pom_path)

            self.assertEqual(pom.modules, [], "Modules should be empty if not present in the POM.")
            self.assertEqual(pom.dependencies, [
                Dependency(group_id="junit", artifact_id="junit", version="3.8.1")
            ])
            self.assertEqual(pom.plugins, [
                Plugin(group_id=None, artifact_id="maven-clean-plugin", version="3.4.0", configuration=None),
                Plugin(group_id=None, artifact_id="maven-site-plugin", version="3.12.1", configuration=None),
                Plugin(group_id=None, artifact_id="maven-project-info-reports-plugin", version="3.6.1", configuration=None),
            ])

    def test_new_pom(self):
        """
        Test that Pom can create a new pom.xml file with the specified group_id and artifact_id.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            artifact_id = "my-app"
            
            pom = Pom.new(workspace, group_id, artifact_id)
            
            self.assertTrue(pom.pom_path.exists(), "New pom.xml should be created.")
            content = pom.pom_path.read_text()
            self.assertIn(f"<groupId>{group_id}</groupId>", content)
            self.assertIn(f"<artifactId>{artifact_id}</artifactId>", content)
    


# class TestPomAddModule(TestCase):
#     """Test cases for Pom.add_module() method."""
    
#     def test_add_single_module(self):
#         """Test adding a single module to the pom.xml."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_module("com.example", "module-a")
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<modules>", modified_pom)
#             self.assertIn("<module>module-a</module>", modified_pom)
    
#     def test_add_multiple_modules(self):
#         """Test adding multiple modules creates separate entries."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_module("com.example", "module-a")
#             proxy.add_module("com.example", "module-b")
#             proxy.add_module("com.example", "module-c")
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertEqual(
#                 modified_pom.count("<module>"),
#                 3,
#                 "There should be exactly 3 <module> entries.",
#             )
#             self.assertIn("<module>module-a</module>", modified_pom)
#             self.assertIn("<module>module-b</module>", modified_pom)
#             self.assertIn("<module>module-c</module>", modified_pom)
    
#     def test_add_module_returns_self_for_chaining(self):
#         """Test that add_module returns self for method chaining."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             result = proxy.add_module("com.example", "module-a")
            
#             self.assertIs(result, proxy, "add_module should return self for chaining.")
    
#     def test_add_duplicate_module_ignored(self):
#         """Test that adding a duplicate module (same artifact_id) is ignored."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_module("com.example", "module-a")
#             proxy.add_module("com.example", "module-a")  # Duplicate
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             # Should only have one module entry
#             self.assertEqual(
#                 modified_pom.count("<module>"),
#                 1,
#                 "Duplicate modules should not be added.",
#             )
            
#             # Cache should also have only one entry
#             self.assertEqual(len(proxy.modules), 1, "Cache should not contain duplicates.")


# class TestPomAddDependency(TestCase):
#     """Test cases for Pom.add_dependency() method."""
    
#     def test_add_dependency_with_version(self):
#         """Test adding a dependency with version specified."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             dependency: Dependency = {
#                 "group_id": "org.junit.jupiter",
#                 "artifact_id": "junit-jupiter-api",
#                 "version": "5.10.0",
#             }
            
#             proxy = Pom(pom_path)
#             proxy.add_dependency(dependency)
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<dependencies>", modified_pom)
#             self.assertIn("<groupId>org.junit.jupiter</groupId>", modified_pom)
#             self.assertIn("<artifactId>junit-jupiter-api</artifactId>", modified_pom)
#             self.assertIn("<version>5.10.0</version>", modified_pom)
    
#     def test_add_dependency_without_version(self):
#         """Test adding a dependency without version (managed dependency)."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             dependency: Dependency = {
#                 "group_id": "org.springframework.boot",
#                 "artifact_id": "spring-boot-starter",
#                 "version": None,
#             }
            
#             proxy = Pom(pom_path)
#             proxy.add_dependency(dependency)
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<groupId>org.springframework.boot</groupId>", modified_pom)
#             self.assertIn("<artifactId>spring-boot-starter</artifactId>", modified_pom)
#             # Should not add empty <version/> tag
#             dep_start = modified_pom.find("<artifactId>spring-boot-starter</artifactId>")
#             dep_end = modified_pom.find("</dependency>", dep_start)
#             dep_section = modified_pom[dep_start:dep_end]
#             self.assertNotIn("<version>", dep_section, "Should not add version tag when version is None.")
    
#     def test_add_multiple_dependencies(self):
#         """Test adding multiple dependencies creates separate entries."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
#             proxy.add_dependency({"group_id": "org.mockito", "artifact_id": "mockito-core", "version": "5.5.0"})
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertEqual(
#                 modified_pom.count("<dependency>"),
#                 2,
#                 "There should be exactly 2 <dependency> entries.",
#             )
    
#     def test_add_dependency_returns_self_for_chaining(self):
#         """Test that add_dependency returns self for method chaining."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             result = proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
            
#             self.assertIs(result, proxy, "add_dependency should return self for chaining.")
    
#     def test_add_duplicate_dependency_updates_version(self):
#         """Test that adding a duplicate dependency updates the version instead of creating a duplicate."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
#             proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "5.10.0"})  # Same group/artifact, different version
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             # Should only have one dependency entry
#             self.assertEqual(
#                 modified_pom.count("<dependency>"),
#                 1,
#                 "Duplicate dependencies should not be added.",
#             )
            
#             # Version should be updated to the new value
#             self.assertIn("<version>5.10.0</version>", modified_pom)
#             self.assertNotIn("<version>4.13.2</version>", modified_pom)
            
#             # Cache should also have only one entry
#             self.assertEqual(len(proxy.dependencies), 1, "Cache should not contain duplicates.")
#             self.assertEqual(proxy.dependencies[0]["version"], "5.10.0", "Cache version should be updated.")


# class TestPomAddPlugin(TestCase):
#     """Test cases for Pom.add_plugin() method."""
    
#     def test_add_plugin_with_version(self):
#         """Test adding a plugin with version to pluginManagement."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             plugin: Plugin = {
#                 "group_id": "com.diffplug.maven",
#                 "artifact_id": "spotless-maven-plugin",
#                 "version": "2.41.0",
#                 "configuration": None,
#             }
            
#             proxy = Pom(pom_path)
#             proxy.add_plugin(plugin)
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<pluginManagement>", modified_pom)
#             self.assertIn("<plugins>", modified_pom)
#             self.assertIn("<groupId>com.diffplug.maven</groupId>", modified_pom)
#             self.assertIn("<artifactId>spotless-maven-plugin</artifactId>", modified_pom)
#             self.assertIn("<version>2.41.0</version>", modified_pom)
    
#     def test_add_plugin_with_configuration(self):
#         """Test adding a plugin with XML configuration."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             plugin: Plugin = {
#                 "group_id": "com.diffplug.maven",
#                 "artifact_id": "spotless-maven-plugin",
#                 "version": "2.41.0",
#                 "configuration": "<java><googleJavaFormat/></java>",
#             }
            
#             proxy = Pom(pom_path)
#             proxy.add_plugin(plugin)
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<configuration>", modified_pom)
#             self.assertIn("<java>", modified_pom)
#             self.assertIn("<googleJavaFormat", modified_pom)
    
#     def test_add_plugin_to_existing_plugin_management(self):
#         """Test that plugins are added to existing pluginManagement section."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(POM_WITH_SECTIONS)
            
#             plugin: Plugin = {
#                 "group_id": "com.diffplug.maven",
#                 "artifact_id": "spotless-maven-plugin",
#                 "version": "2.41.0",
#                 "configuration": None,
#             }
            
#             proxy = Pom(pom_path)
#             proxy.add_plugin(plugin)
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             # Should still have only one pluginManagement section
#             self.assertEqual(
#                 modified_pom.count("<pluginManagement>"),
#                 1,
#                 "There should be exactly one <pluginManagement> section.",
#             )
#             # Should now have 2 plugins (existing + new)
#             self.assertEqual(
#                 modified_pom.count("<plugin>"),
#                 2,
#                 "There should be exactly 2 <plugin> entries in pluginManagement.",
#             )
    
#     def test_add_plugin_returns_self_for_chaining(self):
#         """Test that add_plugin returns self for method chaining."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             plugin: Plugin = {
#                 "group_id": "org.apache.maven.plugins",
#                 "artifact_id": "maven-compiler-plugin",
#                 "version": "3.11.0",
#                 "configuration": None,
#             }
#             result = proxy.add_plugin(plugin)
            
#             self.assertIs(result, proxy, "add_plugin should return self for chaining.")
    
#     def test_add_duplicate_plugin_updates_version_and_config(self):
#         """Test that adding a duplicate plugin updates version and configuration instead of creating a duplicate."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
            
#             # Add plugin with initial version
#             proxy.add_plugin({
#                 "group_id": "com.diffplug.maven",
#                 "artifact_id": "spotless-maven-plugin",
#                 "version": "2.41.0",
#                 "configuration": None,
#             })
            
#             # Add same plugin with new version and configuration
#             proxy.add_plugin({
#                 "group_id": "com.diffplug.maven",
#                 "artifact_id": "spotless-maven-plugin",
#                 "version": "2.43.0",
#                 "configuration": "<java><googleJavaFormat/></java>",
#             })
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             # Should only have one plugin entry in pluginManagement
#             pm_start = modified_pom.find("<pluginManagement>")
#             pm_end = modified_pom.find("</pluginManagement>")
#             pm_section = modified_pom[pm_start:pm_end]
            
#             self.assertEqual(
#                 pm_section.count("<plugin>"),
#                 1,
#                 "Duplicate plugins should not be added.",
#             )
            
#             # Version should be updated to the new value
#             self.assertIn("<version>2.43.0</version>", modified_pom)
#             self.assertNotIn("<version>2.41.0</version>", modified_pom)
            
#             # Configuration should be added
#             self.assertIn("<configuration>", modified_pom)
#             self.assertIn("<googleJavaFormat", modified_pom)
            
#             # Cache should also have only one entry
#             self.assertEqual(len(proxy.plugins), 1, "Cache should not contain duplicates.")
#             self.assertEqual(proxy.plugins[0]["version"], "2.43.0", "Cache version should be updated.")
#             self.assertEqual(proxy.plugins[0]["configuration"], "<java><googleJavaFormat/></java>", "Cache configuration should be updated.")


# class TestPomSetPackaging(TestCase):
#     """Test cases for Pom.set_packaging() method."""
    
#     def test_set_packaging_creates_element(self):
#         """Test that set_packaging creates <packaging> element if it doesn't exist."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.set_packaging("pom")
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<packaging>pom</packaging>", modified_pom)
    
#     def test_set_packaging_updates_existing_element(self):
#         """Test that set_packaging updates existing <packaging> element."""
#         pom_with_packaging = BASE_POM_FOR_PROXY.replace(
#             "</version>",
#             "</version>\n  <packaging>jar</packaging>",
#         )
        
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(pom_with_packaging)
            
#             proxy = Pom(pom_path)
#             proxy.set_packaging("pom")
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<packaging>pom</packaging>", modified_pom)
#             self.assertNotIn("<packaging>jar</packaging>", modified_pom)
    
#     def test_set_packaging_returns_self_for_chaining(self):
#         """Test that set_packaging returns self for method chaining."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             result = proxy.set_packaging("pom")
            
#             self.assertIs(result, proxy, "set_packaging should return self for chaining.")


# class TestPomSave(TestCase):
#     """Test cases for Pom.save() method."""
    
#     def test_save_writes_changes_to_file(self):
#         """Test that save() persists all changes to the pom.xml file."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy.add_module("com.example", "test-module")
#             proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
#             proxy.save()
            
#             modified_pom = pom_path.read_text()
            
#             self.assertIn("<module>test-module</module>", modified_pom)
#             self.assertIn("<groupId>junit</groupId>", modified_pom)
    
#     def test_save_without_root_raises_error(self):
#         """Test that save() raises ValueError if _root is None."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             proxy._root = None  # Simulate invalid state
            
#             with self.assertRaises(ValueError) as context:
#                 proxy.save()
            
#             self.assertIn("root element is None", str(context.exception))


# class TestPomMethodChaining(TestCase):
#     """Test cases for fluent API method chaining."""
    
#     def test_full_chaining_example(self):
#         """Test complete method chaining scenario."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             pom_path = Path(temp_dir, "pom.xml")
#             pom_path.write_text(BASE_POM_FOR_PROXY)
            
#             proxy = Pom(pom_path)
#             (
#                 proxy
#                 .set_packaging("pom")
#                 .add_module("com.example", "module-core")
#                 .add_module("com.example", "module-api")
#                 .add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
#                 .add_plugin({
#                     "group_id": "org.apache.maven.plugins",
#                     "artifact_id": "maven-compiler-plugin",
#                     "version": "3.11.0",
#                     "configuration": None,
#                 })
#                 .save()
#             )
            
#             modified_pom = pom_path.read_text()
#             logging.debug(f"Modified POM:\n{modified_pom}")
            
#             self.assertIn("<packaging>pom</packaging>", modified_pom)
#             self.assertIn("<module>module-core</module>", modified_pom)
#             self.assertIn("<module>module-api</module>", modified_pom)
#             self.assertIn("<artifactId>junit</artifactId>", modified_pom)
#             self.assertIn("<artifactId>maven-compiler-plugin</artifactId>", modified_pom)


# class TestPomBaseFactory(TestCase):
#     """Test cases for Pom.base() factory method."""
    
#     def test_base_creates_pom_in_workspace_root(self):
#         """Test that base() creates pom.xml in the workspace root directory."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "de.hofuniversity"
            
#             proxy = Pom.base(workspace, group_id)
            
#             # Check that pom.xml is created at workspace/pom.xml
#             pom_path = workspace / "pom.xml"
#             self.assertTrue(pom_path.exists(), "pom.xml should be created in workspace root.")
#             self.assertEqual(proxy.pom_path, pom_path, "Proxy should point to workspace/pom.xml.")
    
#     def test_base_creates_minimal_pom_with_group_id(self):
#         """Test that base() creates a pom.xml with the specified group_id."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "de.hofuniversity"
            
#             Pom.base(workspace, group_id)
            
#             content = (workspace / "pom.xml").read_text()
#             self.assertIn(f"<groupId>{group_id}</groupId>", content)
#             self.assertIn("<artifactId>workspace</artifactId>", content)
#             self.assertIn("<version>1.0</version>", content)
#             self.assertIn("<packaging>pom</packaging>", content)
    
#     def test_base_returns_configured_proxy(self):
#         """Test that base() returns a configured Pom instance."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "com.example"
            
#             proxy = Pom.base(workspace, group_id)
            
#             self.assertIsInstance(proxy, Pom)
#             self.assertEqual(proxy.pom_path, workspace / "pom.xml")
#             self.assertIsNotNone(proxy._root)
    
#     def test_base_allows_modifications(self):
#         """Test that the proxy returned by base() can be modified."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "de.hofuniversity"
            
#             proxy = Pom.base(workspace, group_id)
#             proxy.add_module(group_id, "module-a")
#             proxy.save()
            
#             content = (workspace / "pom.xml").read_text()
#             self.assertIn(f"<groupId>{group_id}</groupId>", content)
#             self.assertIn("<module>module-a</module>", content)
    
#     def test_base_creates_workspace_directory(self):
#         """Test that base() creates the workspace directory if it doesn't exist."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir) / "nested" / "workspace"
#             group_id = "com.example"
            
#             proxy = Pom.base(workspace, group_id)
            
#             self.assertTrue(workspace.exists(), "Workspace directory should be created.")
#             self.assertTrue((workspace / "pom.xml").exists(), "pom.xml should be created.")


# class TestPomNewFactory(TestCase):
#     """Test cases for Pom.new() factory method."""
    
#     @classmethod
#     def setUpClass(cls):
#         """Check if Maven is available before running tests."""
#         import shutil
#         cls.maven_available = shutil.which("mvn") is not None
    
#     def setUp(self):
#         """Skip tests if Maven is not available."""
#         if not self.maven_available:
#             self.skipTest("Maven is not installed or not in PATH.")
    
#     def test_new_creates_maven_project_with_simple_archetype(self):
#         """Test that new() creates a Maven project using the simple archetype."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "com.example"
#             artifact_id = "test-app"
            
#             proxy = Pom.new(
#                 workspace=workspace,
#                 group_id=group_id,
#                 artifact_id=artifact_id,
#             )
            
#             # Check that the project was created
#             project_path = workspace / artifact_id
#             pom_path = project_path / "pom.xml"
            
#             self.assertTrue(project_path.exists(), "Project directory should be created.")
#             self.assertTrue(pom_path.exists(), "pom.xml should be created.")
            
#             # Check that the proxy points to the correct pom.xml
#             self.assertEqual(proxy.pom_path, pom_path)
    
#     def test_new_parses_existing_content(self):
#         """Test that new() returns a proxy that has parsed existing pom.xml content."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "com.example"
#             artifact_id = "test-app"
            
#             proxy = Pom.new(
#                 workspace=workspace,
#                 group_id=group_id,
#                 artifact_id=artifact_id,
#             )
            
#             # The simple archetype creates a junit dependency
#             self.assertGreater(len(proxy.dependencies), 0, "Should have parsed dependencies.")
    
#     def test_new_with_custom_version(self):
#         """Test that new() respects custom version parameter."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "com.example"
#             artifact_id = "test-app"
#             version = "2.0.0-RELEASE"
            
#             proxy = Pom.new(
#                 workspace=workspace,
#                 group_id=group_id,
#                 artifact_id=artifact_id,
#                 version=version,
#             )
            
#             pom_path = workspace / artifact_id / "pom.xml"
#             content = pom_path.read_text()
            
#             self.assertIn(f"<version>{version}</version>", content)
    
#     def test_new_allows_further_modifications(self):
#         """Test that the proxy returned by new() can be modified."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             workspace = Path(temp_dir)
#             group_id = "com.example"
#             artifact_id = "test-app"
            
#             proxy = Pom.new(
#                 workspace=workspace,
#                 group_id=group_id,
#                 artifact_id=artifact_id,
#             )
            
#             # Add additional dependency
#             proxy.add_dependency({
#                 "group_id": "org.mockito",
#                 "artifact_id": "mockito-core",
#                 "version": "5.5.0",
#             })
#             proxy.save()
            
#             pom_path = workspace / artifact_id / "pom.xml"
#             content = pom_path.read_text()
            
#             self.assertIn("<artifactId>mockito-core</artifactId>", content)


# class TestPomDeprecatedFunctionsWarning(TestCase):
#     """Test that deprecated functions still work but may show warnings."""
    
#     def test_deprecated_functions_exist(self):
#         """Verify deprecated functions are still available for backwards compatibility."""
#         # These should not raise import errors
#         from mdeagent.preparation.pom import (
#             add_dependencies_to_pom,
#             add_module_to_pom,
#             add_plugin_to_pom,
#         )
        
#         self.assertTrue(callable(add_module_to_pom))
#         self.assertTrue(callable(add_dependencies_to_pom))
#         self.assertTrue(callable(add_plugin_to_pom))
