import logging
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import warnings

from mdeagent.preparation.pom import (
    Dependency,
    Module,
    Plugin,
    PomProxy,
    add_dependencies_to_pom,
    add_module_to_pom,
    add_plugin_to_pom,
    install_dependencies,
)

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
        
        <plugin>
          <artifactId>maven-resources-plugin</artifactId>
          <version>3.3.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.13.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-surefire-plugin</artifactId>
          <version>3.3.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-jar-plugin</artifactId>
          <version>3.4.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-install-plugin</artifactId>
          <version>3.1.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-deploy-plugin</artifactId>
          <version>3.1.2</version>
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


class TestAddDependencies(TestCase):
    def test_add_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM)

            dependencies: list[Dependency] = [
                {
                    "group_id": "org.springframework",
                    "artifact_id": "spring-core",
                    "version": "5.3.8",
                },
                {
                    "group_id": "org.apache.commons",
                    "artifact_id": "commons-lang3",
                    "version": None,
                },
            ]

            add_dependencies_to_pom(pom_path, dependencies)

            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<dependencies>"),
                1,
                "There should be exactly 1 <dependencies> section in the modified POM.",
            )
            self.assertEqual(
                modified_pom.count("<dependency>"),
                2,
                "There should be exactly 2 <dependency> entries in the modified POM.",
            )

            self.assertIn("<dependencies>", modified_pom)
            self.assertIn("<dependency>", modified_pom)
            self.assertIn("<groupId>org.springframework</groupId>", modified_pom)
            self.assertIn("<artifactId>spring-core</artifactId>", modified_pom)
            self.assertIn("<version>5.3.8</version>", modified_pom)
            self.assertIn("<groupId>org.apache.commons</groupId>", modified_pom)
            self.assertIn("<artifactId>commons-lang3</artifactId>", modified_pom)

    def test_add_dependencies_to_existing_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM_WITH_DEPENDENCIES)

            dependencies: list[Dependency] = [
                {
                    "group_id": "org.springframework",
                    "artifact_id": "spring-core",
                    "version": "5.3.8",
                },
                {
                    "group_id": "org.apache.commons",
                    "artifact_id": "commons-lang3",
                    "version": None,
                },
            ]

            add_dependencies_to_pom(pom_path, dependencies)

            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<dependencies>"),
                1,
                "There should be exactly 1 <dependencies> section in the modified POM.",
            )
            self.assertEqual(
                modified_pom.count("<dependency>"),
                3,
                "There should be exactly 3 <dependency> entries in the modified POM.",
            )
            self.assertIn("<groupId>org.springframework</groupId>", modified_pom)
            self.assertIn("<artifactId>spring-core</artifactId>", modified_pom)
            self.assertIn("<version>5.3.8</version>", modified_pom)
            self.assertIn("<groupId>org.apache.commons</groupId>", modified_pom)
            self.assertIn("<artifactId>commons-lang3</artifactId>", modified_pom)


class TestInstallDependencies(TestCase):
    @patch("subprocess.run")
    def test_install_dependencies(self, mock_run: Mock):
        mock_run.return_value.returncode = 0
        workspace_path = Path("/fake/workspace")

        try:
            install_dependencies(workspace_path)
        except RuntimeError:
            self.fail("install_dependencies raised RuntimeError unexpectedly!")

        mock_run.assert_called_once_with(
            ["mvn", "validate"], cwd=workspace_path, check=True
        )


class TestAddModuleToPom(TestCase):
    def test_add_module_to_pom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM)

            group_id = "com.example"
            artifact_id = "my-module"
            version = "1.0-SNAPSHOT"

            add_module_to_pom(pom_path, group_id, artifact_id, version)

            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")

            self.assertIn("<modules>", modified_pom)
            self.assertIn(f"<module>{artifact_id}</module>", modified_pom)


class TestAddPluginToPom(TestCase):
    def test_add_plugin_to_pom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM)

            plugin: Plugin = {
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.x.x",
                "configuration": "<java><googleJavaFormat/></java>",
            }

            add_plugin_to_pom(pom_path, plugin)

            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<plugins>"),
                1,
                "There should be exactly 1 <plugins> section in the modified POM.",
            )
            self.assertEqual(
                modified_pom.count("<plugin>"),
                1,
                "There should be exactly 1 <plugin> entry in the modified POM.",
            )

            self.assertIn("<build>", modified_pom)
            self.assertIn("<plugins>", modified_pom)
            self.assertIn("<plugin>", modified_pom)
            self.assertIn("<groupId>com.diffplug.maven</groupId>", modified_pom)
            self.assertIn(
                "<artifactId>spotless-maven-plugin</artifactId>", modified_pom
            )
            self.assertIn("<version>2.x.x</version>", modified_pom)
            self.assertIn("<configuration>", modified_pom)
            self.assertIn("<java>", modified_pom)
            self.assertIn("<googleJavaFormat", modified_pom)

    def test_add_plugin_to_existing_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM_WITH_DEPENDENCIES)

            plugin: Plugin = {
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.x.x",
                "configuration": "<java><googleJavaFormat/></java>",
            }

            add_plugin_to_pom(pom_path, plugin)

            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")

            self.assertEqual(
                modified_pom.count("<build>"),
                1,
                "There should be exactly 1 <build> section in the modified POM.",
            )
            self.assertEqual(
                modified_pom.count("<plugin>"),
                11,
                "There should be exactly 11 <plugin> entries in the modified POM (10 existing + 1 new).",
            )
            self.assertIn("<groupId>com.diffplug.maven</groupId>", modified_pom)
            self.assertIn(
                "<artifactId>spotless-maven-plugin</artifactId>", modified_pom
            )
            self.assertIn("<version>2.x.x</version>", modified_pom)
            self.assertIn("<configuration>", modified_pom)
            self.assertIn("<java>", modified_pom)
            self.assertIn("<googleJavaFormat", modified_pom)


# =============================================================================
# Tests for PomProxy (New Proxy Pattern Implementation)
# =============================================================================

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
  
  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.13.2</version>
    </dependency>
  </dependencies>
  
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


class TestPomProxyInitialization(TestCase):
    """Test cases for PomProxy initialization and setup."""
    
    def test_init_creates_missing_sections(self):
        """Test that PomProxy creates missing <modules>, <dependencies>, and <plugins> sections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            # Check for self-closing tags (format may vary: <modules/> or <modules />)
            self.assertTrue(
                "<modules/>" in modified_pom or "<modules />" in modified_pom,
                "<modules> section should be created.",
            )
            self.assertTrue(
                "<dependencies/>" in modified_pom or "<dependencies />" in modified_pom,
                "<dependencies> section should be created.",
            )
            self.assertTrue(
                "<plugins/>" in modified_pom or "<plugins />" in modified_pom,
                "<plugins> section should be created within pluginManagement.",
            )
    
    def test_init_uses_existing_sections(self):
        """Test that PomProxy reuses existing sections instead of creating duplicates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(POM_WITH_SECTIONS)
            
            proxy = PomProxy(pom_path)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            # Count occurrences - should only have one of each section
            self.assertEqual(
                modified_pom.count("<dependencies>"),
                1,
                "There should be exactly one <dependencies> section.",
            )
            self.assertEqual(
                modified_pom.count("<pluginManagement>"),
                1,
                "There should be exactly one <pluginManagement> section.",
            )


class TestPomProxyAddModule(TestCase):
    """Test cases for PomProxy.add_module() method."""
    
    def test_add_single_module(self):
        """Test adding a single module to the pom.xml."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.add_module("com.example", "module-a")
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<modules>", modified_pom)
            self.assertIn("<module>module-a</module>", modified_pom)
    
    def test_add_multiple_modules(self):
        """Test adding multiple modules creates separate entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.add_module("com.example", "module-a")
            proxy.add_module("com.example", "module-b")
            proxy.add_module("com.example", "module-c")
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
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
            
            proxy = PomProxy(pom_path)
            result = proxy.add_module("com.example", "module-a")
            
            self.assertIs(result, proxy, "add_module should return self for chaining.")


class TestPomProxyAddDependency(TestCase):
    """Test cases for PomProxy.add_dependency() method."""
    
    def test_add_dependency_with_version(self):
        """Test adding a dependency with version specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            dependency: Dependency = {
                "group_id": "org.junit.jupiter",
                "artifact_id": "junit-jupiter-api",
                "version": "5.10.0",
            }
            
            proxy = PomProxy(pom_path)
            proxy.add_dependency(dependency)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<dependencies>", modified_pom)
            self.assertIn("<groupId>org.junit.jupiter</groupId>", modified_pom)
            self.assertIn("<artifactId>junit-jupiter-api</artifactId>", modified_pom)
            self.assertIn("<version>5.10.0</version>", modified_pom)
    
    def test_add_dependency_without_version(self):
        """Test adding a dependency without version (managed dependency)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            dependency: Dependency = {
                "group_id": "org.springframework.boot",
                "artifact_id": "spring-boot-starter",
                "version": None,
            }
            
            proxy = PomProxy(pom_path)
            proxy.add_dependency(dependency)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<groupId>org.springframework.boot</groupId>", modified_pom)
            self.assertIn("<artifactId>spring-boot-starter</artifactId>", modified_pom)
            # Should not add empty <version/> tag
            dep_start = modified_pom.find("<artifactId>spring-boot-starter</artifactId>")
            dep_end = modified_pom.find("</dependency>", dep_start)
            dep_section = modified_pom[dep_start:dep_end]
            self.assertNotIn("<version>", dep_section, "Should not add version tag when version is None.")
    
    def test_add_multiple_dependencies(self):
        """Test adding multiple dependencies creates separate entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
            proxy.add_dependency({"group_id": "org.mockito", "artifact_id": "mockito-core", "version": "5.5.0"})
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
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
            
            proxy = PomProxy(pom_path)
            result = proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
            
            self.assertIs(result, proxy, "add_dependency should return self for chaining.")


class TestPomProxyAddPlugin(TestCase):
    """Test cases for PomProxy.add_plugin() method."""
    
    def test_add_plugin_with_version(self):
        """Test adding a plugin with version to pluginManagement."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            plugin: Plugin = {
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.41.0",
                "configuration": None,
            }
            
            proxy = PomProxy(pom_path)
            proxy.add_plugin(plugin)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<pluginManagement>", modified_pom)
            self.assertIn("<plugins>", modified_pom)
            self.assertIn("<groupId>com.diffplug.maven</groupId>", modified_pom)
            self.assertIn("<artifactId>spotless-maven-plugin</artifactId>", modified_pom)
            self.assertIn("<version>2.41.0</version>", modified_pom)
    
    def test_add_plugin_with_configuration(self):
        """Test adding a plugin with XML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            plugin: Plugin = {
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.41.0",
                "configuration": "<java><googleJavaFormat/></java>",
            }
            
            proxy = PomProxy(pom_path)
            proxy.add_plugin(plugin)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<configuration>", modified_pom)
            self.assertIn("<java>", modified_pom)
            self.assertIn("<googleJavaFormat", modified_pom)
    
    def test_add_plugin_to_existing_plugin_management(self):
        """Test that plugins are added to existing pluginManagement section."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(POM_WITH_SECTIONS)
            
            plugin: Plugin = {
                "group_id": "com.diffplug.maven",
                "artifact_id": "spotless-maven-plugin",
                "version": "2.41.0",
                "configuration": None,
            }
            
            proxy = PomProxy(pom_path)
            proxy.add_plugin(plugin)
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
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
            
            proxy = PomProxy(pom_path)
            plugin: Plugin = {
                "group_id": "org.apache.maven.plugins",
                "artifact_id": "maven-compiler-plugin",
                "version": "3.11.0",
                "configuration": None,
            }
            result = proxy.add_plugin(plugin)
            
            self.assertIs(result, proxy, "add_plugin should return self for chaining.")


class TestPomProxySetPackaging(TestCase):
    """Test cases for PomProxy.set_packaging() method."""
    
    def test_set_packaging_creates_element(self):
        """Test that set_packaging creates <packaging> element if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.set_packaging("pom")
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
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
            
            proxy = PomProxy(pom_path)
            proxy.set_packaging("pom")
            proxy.save()
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<packaging>pom</packaging>", modified_pom)
            self.assertNotIn("<packaging>jar</packaging>", modified_pom)
    
    def test_set_packaging_returns_self_for_chaining(self):
        """Test that set_packaging returns self for method chaining."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            result = proxy.set_packaging("pom")
            
            self.assertIs(result, proxy, "set_packaging should return self for chaining.")


class TestPomProxySave(TestCase):
    """Test cases for PomProxy.save() method."""
    
    def test_save_writes_changes_to_file(self):
        """Test that save() persists all changes to the pom.xml file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy.add_module("com.example", "test-module")
            proxy.add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
            proxy.save()
            
            modified_pom = pom_path.read_text()
            
            self.assertIn("<module>test-module</module>", modified_pom)
            self.assertIn("<groupId>junit</groupId>", modified_pom)
    
    def test_save_without_root_raises_error(self):
        """Test that save() raises ValueError if _root is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            proxy._root = None  # Simulate invalid state
            
            with self.assertRaises(ValueError) as context:
                proxy.save()
            
            self.assertIn("root element is None", str(context.exception))


class TestPomProxyMethodChaining(TestCase):
    """Test cases for fluent API method chaining."""
    
    def test_full_chaining_example(self):
        """Test complete method chaining scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(BASE_POM_FOR_PROXY)
            
            proxy = PomProxy(pom_path)
            (
                proxy
                .set_packaging("pom")
                .add_module("com.example", "module-core")
                .add_module("com.example", "module-api")
                .add_dependency({"group_id": "junit", "artifact_id": "junit", "version": "4.13.2"})
                .add_plugin({
                    "group_id": "org.apache.maven.plugins",
                    "artifact_id": "maven-compiler-plugin",
                    "version": "3.11.0",
                    "configuration": None,
                })
                .save()
            )
            
            modified_pom = pom_path.read_text()
            logging.debug(f"Modified POM:\n{modified_pom}")
            
            self.assertIn("<packaging>pom</packaging>", modified_pom)
            self.assertIn("<module>module-core</module>", modified_pom)
            self.assertIn("<module>module-api</module>", modified_pom)
            self.assertIn("<artifactId>junit</artifactId>", modified_pom)
            self.assertIn("<artifactId>maven-compiler-plugin</artifactId>", modified_pom)


class TestPomProxyDeprecatedFunctionsWarning(TestCase):
    """Test that deprecated functions still work but may show warnings."""
    
    def test_deprecated_functions_exist(self):
        """Verify deprecated functions are still available for backwards compatibility."""
        # These should not raise import errors
        from mdeagent.preparation.pom import (
            add_module_to_pom,
            add_dependencies_to_pom,
            add_plugin_to_pom,
        )
        
        self.assertTrue(callable(add_module_to_pom))
        self.assertTrue(callable(add_dependencies_to_pom))
        self.assertTrue(callable(add_plugin_to_pom))
