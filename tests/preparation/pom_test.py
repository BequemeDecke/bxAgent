from typing import List
from unittest import TestCase

from bxagent.preparation.pom import Dependency, add_dependencies_to_pom

INITIAL_POM = """<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>com.example</groupId>
    <artifactId>my-app</artifactId>
    <version>1.0-SNAPSHOT</version>
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
