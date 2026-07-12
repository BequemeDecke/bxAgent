from pathlib import Path
from typing import List
from unittest import TestCase
from unittest.mock import Mock, patch

from bxagent.preparation.pom import Dependency, add_dependencies_to_pom, install_dependencies

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

        modified_pom = add_dependencies_to_pom(INITIAL_POM, dependencies)

        self.assertIn("<dependencies>", modified_pom)
        self.assertIn("<dependency>", modified_pom)
        self.assertIn("<groupId>org.springframework</groupId>", modified_pom)
        self.assertIn("<artifactId>spring-core</artifactId>", modified_pom)
        self.assertIn("<version>5.3.8</version>", modified_pom)
        self.assertIn("<groupId>org.apache.commons</groupId>", modified_pom)
        self.assertIn("<artifactId>commons-lang3</artifactId>", modified_pom)

    def test_add_dependencies_to_existing_dependencies(self):
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

        modified_pom = add_dependencies_to_pom(
            INITIAL_POM_WITH_DEPENDENCIES, dependencies
        )

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