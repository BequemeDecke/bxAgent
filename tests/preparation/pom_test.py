import logging
import tempfile
from pathlib import Path
from typing import List
from unittest import TestCase
from unittest.mock import Mock, patch

from bxagent.preparation.pom import (
    Dependency,
    add_dependencies_to_pom,
    add_module_to_pom,
    install_dependencies,
)

INITIAL_POM = """<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>"""

INITIAL_POM_WITH_DEPENDENCIES = """<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>
    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-core</artifactId>
            <version>5.3.8</version>
        </dependency>
        <dependency>
            <groupId>org.apache.commons</groupId>
            <artifactId>commons-lang3</artifactId>
        </dependency>
    </dependencies>
</project>"""


class TestAddDependencies(TestCase):
    def test_add_dependencies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pom_path = Path(temp_dir, "pom.xml")
            pom_path.write_text(INITIAL_POM)

            dependencies: List[Dependency] = [
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

            dependencies: List[Dependency] = [
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

            self.assertIn("<project>", modified_pom)
            self.assertIn("<dependencies>", modified_pom)
            self.assertEqual(modified_pom.count("<dependency>"), 4)
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