import logging
import tempfile
from pathlib import Path
from unittest import TestCase

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

BASE_POM_FOR_PROXY = """<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-project</artifactId>
  <version>1.0-SNAPSHOT</version>
</project>"""

POM_WITH_SECTIONS = """<?xml version='1.0' encoding='utf-8'?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>test-project</artifactId>
  <version>1.0-SNAPSHOT</version>
  <build>
    <pluginManagement>
      <plugins>
        <plugin>
          <groupId>org.apache.maven.plugins</groupId>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.11.0</version>
        </plugin>
      </plugins>
    </pluginManagement>
  </build>
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

            self.assertEqual(
                pom.modules, [], "Modules should be empty if not present in the POM."
            )
            self.assertEqual(
                pom.dependencies,
                [Dependency(group_id="junit", artifact_id="junit", version="3.8.1")],
            )
            self.assertEqual(
                pom.plugins,
                [
                    Plugin(
                        group_id=None,
                        artifact_id="maven-clean-plugin",
                        version="3.4.0",
                        configuration=None,
                    ),
                    Plugin(
                        group_id=None,
                        artifact_id="maven-site-plugin",
                        version="3.12.1",
                        configuration=None,
                    ),
                    Plugin(
                        group_id=None,
                        artifact_id="maven-project-info-reports-plugin",
                        version="3.6.1",
                        configuration=None,
                    ),
                ],
            )

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


class TestPomAddModule(TestCase):
    """Test cases for Pom.add_module() method."""

    def test_add_single_module(self):
        """Test adding a single module to the pom.xml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            module = Module(artifact_id="module-a")
            pom.add_module(module)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<modules>", modified_pom)
            self.assertIn("<module>module-a</module>", modified_pom)

    def test_add_multiple_modules(self):
        """Test adding multiple modules creates separate entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            pom.add_module(Module(artifact_id="module-a"))
            pom.add_module(Module(artifact_id="module-b"))
            pom.add_module(Module(artifact_id="module-c"))
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<module>"),
                3,
                "There should be exactly 3 <module> entries.",
            )
            self.assertIn("<module>module-a</module>", modified_pom)
            self.assertIn("<module>module-b</module>", modified_pom)
            self.assertIn("<module>module-c</module>", modified_pom)

    def test_add_module_returns_self_for_chaining(self):
        """Test that add_module returns self for method chaining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            result = pom.add_module(Module(artifact_id="module-a"))

            self.assertIs(result, pom, "add_module should return self for chaining.")

    def test_add_duplicate_module_ignored(self):
        """Test that adding a duplicate module (same artifact_id) is ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            pom.add_module(Module(artifact_id="module-a"))
            pom.add_module(Module(artifact_id="module-a"))  # Duplicate
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            # Should only have one module entry
            self.assertEqual(
                modified_pom.count("<module>"),
                1,
                "Duplicate modules should not be added.",
            )

            # Cache should also have only one entry
            self.assertEqual(
                len(pom.modules), 1, "Cache should not contain duplicates."
            )


class TestPomAddDependency(TestCase):
    """Test cases for Pom.add_dependency() method."""

    def test_add_dependency_with_version(self):
        """Test adding a dependency with version specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            dependency = Dependency(
                group_id="org.junit.jupiter",
                artifact_id="junit-jupiter-api",
                version="5.10.0",
            )

            pom = Pom(pom_path)
            pom.add_dependency(dependency)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<dependencies>", modified_pom)
            self.assertIn("<groupId>org.junit.jupiter</groupId>", modified_pom)
            self.assertIn("<artifactId>junit-jupiter-api</artifactId>", modified_pom)
            self.assertIn("<version>5.10.0</version>", modified_pom)

    def test_add_dependency_without_version(self):
        """Test adding a dependency without version (managed dependency)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            dependency = Dependency(
                group_id="org.springframework.boot",
                artifact_id="spring-boot-starter",
                version=None,
            )

            pom = Pom(pom_path)
            pom.add_dependency(dependency)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<groupId>org.springframework.boot</groupId>", modified_pom)
            self.assertIn("<artifactId>spring-boot-starter</artifactId>", modified_pom)
            # Should not add empty <version/> tag
            dep_start = modified_pom.find(
                "<artifactId>spring-boot-starter</artifactId>"
            )
            dep_end = modified_pom.find("</dependency>", dep_start)
            dep_section = modified_pom[dep_start:dep_end]
            self.assertNotIn(
                "<version>",
                dep_section,
                "Should not add version tag when version is None.",
            )

    def test_add_multiple_dependencies(self):
        """Test adding multiple dependencies creates separate entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            pom.add_dependency(Dependency("junit", "junit", "4.13.2"))
            pom.add_dependency(Dependency("org.mockito", "mockito-core", "5.5.0"))
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<dependency>"),
                2,
                "There should be exactly 2 <dependency> entries.",
            )

    def test_add_dependency_returns_self_for_chaining(self):
        """Test that add_dependency returns self for method chaining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            result = pom.add_dependency(Dependency("junit", "junit", "4.13.2"))

            self.assertIs(
                result, pom, "add_dependency should return self for chaining."
            )

    def test_add_duplicate_dependency_updates_version(self):
        """Test that adding a duplicate dependency updates the version instead of creating a duplicate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            pom.add_dependency(Dependency("junit", "junit", "4.13.2"))
            pom.add_dependency(
                Dependency("junit", "junit", "5.10.0")
            )  # Same group/artifact, different version
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            # Should only have one dependency entry
            self.assertEqual(
                modified_pom.count("<dependency>"),
                1,
                "Duplicate dependencies should not be added.",
            )

            # Version should be updated to the new value
            self.assertIn("<version>5.10.0</version>", modified_pom)
            self.assertNotIn("<version>4.13.2</version>", modified_pom)

            # Cache should also have only one entry
            self.assertEqual(
                len(pom.dependencies), 1, "Cache should not contain duplicates."
            )
            self.assertEqual(
                pom.dependencies[0].version,
                "5.10.0",
                "Cache version should be updated.",
            )


class TestPomAddPlugin(TestCase):
    """Test cases for Pom.add_plugin() method."""

    def test_add_plugin_with_version(self):
        """Test adding a plugin with version to pluginManagement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            plugin = Plugin(
                group_id="com.diffplug.maven",
                artifact_id="spotless-maven-plugin",
                version="2.41.0",
                configuration=None,
            )

            pom = Pom(pom_path)
            pom.add_plugin(plugin)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<pluginManagement>", modified_pom)
            self.assertIn("<plugins>", modified_pom)
            self.assertIn("<groupId>com.diffplug.maven</groupId>", modified_pom)
            self.assertIn(
                "<artifactId>spotless-maven-plugin</artifactId>", modified_pom
            )
            self.assertIn("<version>2.41.0</version>", modified_pom)

    def test_add_plugin_with_configuration(self):
        """Test adding a plugin with XML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            plugin = Plugin(
                group_id="com.diffplug.maven",
                artifact_id="spotless-maven-plugin",
                version="2.41.0",
                configuration="<java><googleJavaFormat/></java>",
            )

            pom = Pom(pom_path)
            pom.add_plugin(plugin)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<configuration>", modified_pom)
            self.assertIn("<java>", modified_pom)
            self.assertIn("<googleJavaFormat", modified_pom)

    def test_add_plugin_to_existing_plugin_management(self):
        """Test that plugins are added to existing pluginManagement section."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(POM_WITH_SECTIONS)

            plugin = Plugin(
                group_id="com.diffplug.maven",
                artifact_id="spotless-maven-plugin",
                version="2.41.0",
                configuration=None,
            )

            pom = Pom(pom_path)
            pom.add_plugin(plugin)
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            # Should still have only one pluginManagement section
            self.assertEqual(
                modified_pom.count("<pluginManagement>"),
                1,
                "There should be exactly one <pluginManagement> section.",
            )
            # Should now have 2 plugins (existing + new)
            self.assertEqual(
                modified_pom.count("<plugin>"),
                2,
                "There should be exactly 2 <plugin> entries in pluginManagement.",
            )

    def test_add_plugin_returns_self_for_chaining(self):
        """Test that add_plugin returns self for method chaining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            plugin = Plugin(
                group_id="org.apache.maven.plugins",
                artifact_id="maven-compiler-plugin",
                version="3.11.0",
                configuration=None,
            )
            result = pom.add_plugin(plugin)

            self.assertIs(result, pom, "add_plugin should return self for chaining.")

    def test_add_duplicate_plugin_updates_version_and_config(self):
        """Test that adding a duplicate plugin updates version and configuration instead of creating a duplicate."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)

            # Add plugin with initial version
            pom.add_plugin(
                Plugin(
                    group_id="com.diffplug.maven",
                    artifact_id="spotless-maven-plugin",
                    version="2.41.0",
                    configuration=None,
                )
            )

            # Add same plugin with new version and configuration
            pom.add_plugin(
                Plugin(
                    group_id="com.diffplug.maven",
                    artifact_id="spotless-maven-plugin",
                    version="2.43.0",
                    configuration="<java><googleJavaFormat/></java>",
                )
            )
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            # Should only have one plugin entry in pluginManagement
            pm_start = modified_pom.find("<pluginManagement>")
            pm_end = modified_pom.find("</pluginManagement>")
            pm_section = modified_pom[pm_start:pm_end]

            self.assertEqual(
                pm_section.count("<plugin>"),
                1,
                "Duplicate plugins should not be added.",
            )

            # Version should be updated to the new value
            self.assertIn("<version>2.43.0</version>", modified_pom)
            self.assertNotIn("<version>2.41.0</version>", modified_pom)

            # Configuration should be added
            self.assertIn("<configuration>", modified_pom)
            self.assertIn("<googleJavaFormat", modified_pom)

            # Cache should also have only one entry
            self.assertEqual(
                len(pom.plugins), 1, "Cache should not contain duplicates."
            )
            self.assertEqual(
                pom.plugins[0].version, "2.43.0", "Cache version should be updated."
            )
            self.assertEqual(
                pom.plugins[0].configuration,
                "<java><googleJavaFormat/></java>",
                "Cache configuration should be updated.",
            )


class TestPomSetPackaging(TestCase):
    """Test cases for Pom.set_packaging() method."""

    def test_set_packaging_creates_element(self):
        """Test that set_packaging creates <packaging> element if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            pom.set_packaging("pom")
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<packaging>pom</packaging>", modified_pom)

    def test_set_packaging_updates_existing_element(self):
        """Test that set_packaging updates existing <packaging> element."""
        pom_with_packaging = BASE_POM_FOR_PROXY.replace(
            "</version>",
            "</version>\n  <packaging>jar</packaging>",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(pom_with_packaging)

            pom = Pom(pom_path)
            pom.set_packaging("pom")
            pom.save()

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<packaging>pom</packaging>", modified_pom)
            self.assertNotIn("<packaging>jar</packaging>", modified_pom)

    def test_set_packaging_returns_self_for_chaining(self):
        """Test that set_packaging returns self for method chaining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            result = pom.set_packaging("pom")

            self.assertIs(result, pom, "set_packaging should return self for chaining.")


class TestPomSave(TestCase):
    """Test cases for Pom.save() method."""

    def test_save_preserves_existing_structure(self):
        """Test that save() preserves the basic structure even without modifications."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM)

            pom = Pom(pom_path)
            pom.save()

            # Read the saved file and verify essential elements are preserved
            actual_pom = pom_path.read_text()
            self.assertIn("<modelVersion>4.0.0</modelVersion>", actual_pom)
            self.assertIn("<groupId>com.example</groupId>", actual_pom)
            self.assertIn("<artifactId>my-app</artifactId>", actual_pom)
            self.assertIn("<version>1.0-SNAPSHOT</version>", actual_pom)


class TestPomMethodChaining(TestCase):
    """Test cases for fluent API method chaining."""

    def test_full_chaining_example(self):
        """Test complete method chaining scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)

            pom = Pom(pom_path)
            (
                pom.set_packaging("pom")
                .add_module(Module("module-core"))
                .add_module(Module("module-api"))
                .add_dependency(Dependency("junit", "junit", "4.13.2"))
                .add_plugin(
                    Plugin(
                        group_id="org.apache.maven.plugins",
                        artifact_id="maven-compiler-plugin",
                        version="3.11.0",
                        configuration=None,
                    )
                )
                .save()
            )

            modified_pom = pom_path.read_text()
            logger.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<packaging>pom</packaging>", modified_pom)
            self.assertIn("<module>module-core</module>", modified_pom)
            self.assertIn("<module>module-api</module>", modified_pom)
            self.assertIn("<artifactId>junit</artifactId>", modified_pom)
            self.assertIn(
                "<artifactId>maven-compiler-plugin</artifactId>", modified_pom
            )


class TestPomNewFactory(TestCase):
    """Test cases for Pom.new() factory method."""

    def test_new_creates_pom_in_workspace_root(self):
        """Test that new() creates pom.xml in the workspace root directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "de.hofuniversity"
            artifact_id = "workspace"

            pom = Pom.new(workspace, group_id, artifact_id)

            # Check that pom.xml is created at workspace/pom.xml
            pom_path = workspace / "pom.xml"
            self.assertTrue(
                pom_path.exists(), "pom.xml should be created in workspace root."
            )
            self.assertEqual(
                pom.pom_path, pom_path, "Pom should point to workspace/pom.xml."
            )

    def test_new_creates_minimal_pom_with_group_id_and_artifact_id(self):
        """Test that new() creates a pom.xml with the specified group_id and artifact_id."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "de.hofuniversity"
            artifact_id = "workspace"

            Pom.new(workspace, group_id, artifact_id)

            content = (workspace / "pom.xml").read_text()
            self.assertIn(f"<groupId>{group_id}</groupId>", content)
            self.assertIn(f"<artifactId>{artifact_id}</artifactId>", content)
            self.assertIn("<version>1.0</version>", content)
            self.assertIn("<packaging>pom</packaging>", content)

    def test_new_returns_configured_pom(self):
        """Test that new() returns a configured Pom instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "com.example"
            artifact_id = "my-project"

            pom = Pom.new(workspace, group_id, artifact_id)

            self.assertIsInstance(pom, Pom)
            self.assertEqual(pom.pom_path, workspace / "pom.xml")

    def test_new_allows_modifications(self):
        """Test that the Pom returned by new() can be modified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            group_id = "de.hofuniversity"
            artifact_id = "workspace"

            pom = Pom.new(workspace, group_id, artifact_id)
            pom.add_module(Module("module-a"))
            pom.save()

            content = (workspace / "pom.xml").read_text()
            self.assertIn(f"<groupId>{group_id}</groupId>", content)
            self.assertIn("<module>module-a</module>", content)

    def test_new_creates_workspace_directory(self):
        """Test that new() creates the workspace directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "nested" / "workspace"
            group_id = "com.example"
            artifact_id = "my-project"

            pom = Pom.new(workspace, group_id, artifact_id)

            self.assertTrue(
                workspace.exists(), "Workspace directory should be created."
            )
            self.assertTrue(
                (workspace / "pom.xml").exists(), "pom.xml should be created."
            )
