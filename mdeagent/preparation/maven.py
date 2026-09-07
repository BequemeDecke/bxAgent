import subprocess
from pathlib import Path

from mdeagent.preparation.pom import Module, Pom


class MavenProject:
    pom: Pom
    workspace: Path

    def __init__(self, pom: Pom, workspace: Path):
        self.pom = pom
        self.workspace = workspace

    def validate(self) -> bool:
        """
        Execute `mvn validate` to check if the Maven project is valid.

        :return: True if the project is valid, False otherwise.
        """
        validate_process = subprocess.run(
            ["mvn", "validate"],
            check=True,
            cwd=self.workspace
        )
        return validate_process.returncode == 0

    def build(self) -> bool:
        """
        Execute `mvn package` to build the Maven project.

        :return: True if the build was successful, False otherwise.
        """
        build_process = subprocess.run(
            ["mvn", "package"],
            check=True,
            cwd=self.workspace
        )
        return build_process.returncode == 0

    def format_code(self) -> bool:
        """
        Execute `mvn spotless:apply` to format the code in the Maven project.

        :return: True if the formatting was successful, False otherwise.
        """
        format_process = subprocess.run(
            ["mvn", "spotless:apply"],
            check=True,
            cwd=self.workspace
        )
        return format_process.returncode == 0

    def add_file(self, relative_path: Path, content: str) -> None:
        """
        Add a new file to the Maven project.

        :param relative_path: The relative path of the file to be added.
        :param content: The content to write into the file.
        """
        full_path = self.workspace / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    def add_java_class(self, package: str, class_name: str, content: str) -> None:
        """
        Add a new Java class to the Maven project.

        :param package: The package name for the Java class.
        :param class_name: The name of the Java class.
        :param content: The content of the Java class.
        """
        package_path = Path(*package.split('.'))
        java_file_path = package_path / f"{class_name}.java"
        self.add_file(java_file_path, content)

    @classmethod
    def load(cls, workspace: Path) -> "MavenProject":
        """
        Load an existing Maven project from the specified workspace.

        :param workspace: The path to the workspace where the Maven project is located.
        :return: An instance of MavenProject representing the loaded project.
        """
        pom_path = workspace / "pom.xml"
        if not pom_path.exists():
            raise FileNotFoundError(f"pom.xml not found in {workspace}")
        pom = Pom(pom_path)
        return cls(pom, workspace)

    @classmethod
    def create(
        cls, workspace: Path, group_id: str, artifact_id: str, parent: "MavenProject | None"
    ) -> "MavenProject":
        """
        Create a new Maven project in the specified workspace.

        :param workspace: The path to the workspace where the project will be created.
        :param group_id: The group ID for the Maven project.
        :param artifact_id: The artifact ID for the Maven project.
        :param parent: The parent Maven project.
        :return: An instance of MavenProject representing the created project.
        """
        if parent is None:
            pom = Pom.new(workspace, group_id=group_id, artifact_id=artifact_id)
            return cls(pom, workspace)

        cp_process = subprocess.run(
            [
                "mvn",
                "archetype:generate",
                "-DgroupId=" + group_id,
                "-DartifactId=" + artifact_id,
                "-DarchetypeArtifactId=maven-archetype-simple",
                "-DarchetypeVersion=1.5",
                "-DinteractiveMode=false",
            ],
            check=True,
            cwd=workspace,
        )
        if cp_process.returncode != 0:
            raise RuntimeError(
                f"Failed to create Maven project. Return code: {cp_process.returncode}"
            )
        parent.pom.add_module(Module(artifact_id))
        pom_path = workspace / artifact_id / "pom.xml"
        pom = Pom(pom_path)
        return cls(pom, workspace)
