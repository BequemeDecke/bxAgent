from abc import ABC, abstractmethod
from pathlib import Path

from mdeagent.comprehension import FileTransformationPlanParser, TransformationPlan
from mdeagent.config import Config
from mdeagent.preparation.maven import MavenProject
from mdeagent.preparation.pom import Dependency, Module, Plugin, Pom
from mdeagent.preparation.state import PreparationState

EMF_DEPENDENCIES: list[Dependency] = [
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.ecore",
        version="2.30.0",
    ),
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.common",
        version="2.30.0",
    ),
    Dependency(
        group_id="org.eclipse.emf",
        artifact_id="org.eclipse.emf.ecore.xmi",
        version="2.30.0",
    ),
]

# This is tested on Java 23.0.1
SPOTLESS_PLUGIN = Plugin(
    group_id="com.diffplug.spotless",
    artifact_id="spotless-maven-plugin",
    version="3.0.0",
    configuration={
        "java": {
            "includes": {
                "include": ["src/main/java/**/*.java", "src/test/java/**/*.java"]
            },
            "palantirJavaFormat": {"version": "2.71.0"},
        }
    },
)


class StructureFixStrategy(ABC):
    """
    Abstract base class for strategies to fix the workspace structure.
    """

    @abstractmethod
    def fix_structure(self, state: PreparationState) -> PreparationState:
        pass


def is_workspace_structure_correct(
    workspace: Path, group_id: str, artifact_id: str
) -> bool:
    """
    Check if the workspace structure is valid.
    Returns True if the structure is valid, False otherwise.
    """
    # Check for the existence of the parent pom.xml
    parent_pom_path = workspace / "pom.xml"
    if not parent_pom_path.exists():
        return False

    # Check for the existence of the transformation module (Maven project)
    transformation_module_path = workspace / artifact_id
    if not transformation_module_path.exists():
        return False

    # Check for the existence of the TRANSFORMATION.md file
    transformation_md_path = transformation_module_path / "TRANSFORMATION.md"
    if not transformation_md_path.exists():
        return False

    return True


def create_prepare_workspace_node(fix_strategy: StructureFixStrategy):
    def prepare_workspace_node(state: PreparationState) -> PreparationState:
        workspace = state.get("workspace_path")
        if workspace is None:
            raise ValueError("Workspace path is not set in the state.")

        group_id = state.get("group_id")
        if group_id is None:
            raise ValueError("Group ID is not set in the state.")

        artifact_id = state.get("artifact_id")
        if artifact_id is None:
            raise ValueError("Artifact ID is not set in the state.")

        # Create workspace if directory does not exist
        workspace.mkdir(parents=True, exist_ok=True)

        # Check if the folder is empty => Create the Parent project, else execute the strategy
        fixed_state = {}  # State to overwrite
        if not any(workspace.iterdir()):
            # Create the parent Maven project in the workspace
            # artifact_id in parent project must not be the same as the child project
            parent_artifact_id = workspace.name
            parent_project = MavenProject.create(
                workspace, group_id, parent_artifact_id, None
            )
            project = MavenProject.create(
                workspace, group_id, artifact_id, parent_project
            )
        elif any(workspace.iterdir()) and not is_workspace_structure_correct(
            workspace, group_id, artifact_id
        ):
            # Project structure is incorrect, apply the fix strategy
            fixed_state = fix_strategy.fix_structure(state)
            project = MavenProject.load(
                workspace / artifact_id
            )  # Workspace should be fixed
        else:
            # Project structure is correct, load the existing Maven project
            project = MavenProject.load(workspace / artifact_id)

        # Create the transformation module (Maven project) inside the workspace
        # Package path includes artifact_id as subpackage (e.g., de.example.mdeagent)
        full_package = f"{group_id}.{artifact_id}"
        package_path = project.get_package_path(full_package)
        if not package_path.exists():
            package_path.mkdir(parents=True)

        # Create the TRANSFORMATION.md file
        transformation_md_path = workspace / artifact_id / "TRANSFORMATION.md"
        tp_parser = FileTransformationPlanParser(transformation_md_path)
        tp = TransformationPlan.parse(
            parser=tp_parser
        )  # Transformation plan is created if not existing, else loaded

        # Create the transformation Java file (bxtool)
        transformation_class_name = (
            Config.get_instance().VARIABLES.TRANSFORMATION_CLASS_NAME
        )
        transformation_class_path = package_path / f"{transformation_class_name}.java"
        bxtool_path = package_path / f"{transformation_class_name}BxToolAdapter.java"
        if not bxtool_path.exists():
            bxtool_path.touch()

        # Delete the App.java file created by the Maven archetype
        app_java_path = package_path / ".." / "App.java"
        if app_java_path.exists():
            app_java_path.unlink()

        # Add EMF dependencies to the pom.xml of the transformation module
        for dependency in EMF_DEPENDENCIES:
            project.pom.add_dependency(dependency)
        project.pom.add_plugin(SPOTLESS_PLUGIN)
        project.pom.save()
        project.validate()

        # Update the state with the new paths and transformation plan
        new_state = PreparationState(
            transformation_plan=tp,
            bxtool_path=bxtool_path,
            transformation_class_path=transformation_class_path,
            maven_project_path=project.workspace,
        )
        new_state.update(fixed_state)
        return new_state

    return prepare_workspace_node
