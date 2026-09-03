import logging
import tempfile
from pathlib import Path
from typing import list
from unittest import TestCase
from unittest.mock import Mock, patch

from mdeagent.preparation.pom import (
    Dependency,
    Plugin,
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
