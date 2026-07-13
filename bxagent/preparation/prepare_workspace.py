from abc import ABC, abstractmethod
import subprocess

from pathlib import Path
from typing import List

from bxagent.comprehension import TransformationPlan, FileTransformationPlanParser

import bxagent.preparation.pom as pom_utils
from .state import PreparationState


EMF_DEPENDENCIES: List[pom_utils.Dependency] = [
    {
        "group_id": "org.eclipse.emf",
        "artifact_id": "org.eclipse.emf.ecore",
        "version": "2.30.0",
    },
    {
        "group_id": "org.eclipse.emf",
        "artifact_id": "org.eclipse.emf.common",
        "version": "2.30.0",
    },
    {
        "group_id": "org.eclipse.emf",
        "artifact_id": "org.eclipse.emf.ecore.xmi",
        "version": "2.30.0",
    }
]


class StructureFixStrategy(ABC):
    """
    Abstract base class for strategies to fix the workspace structure.
    """

    @abstractmethod
    def fix_structure(self, state: PreparationState) -> PreparationState:
        pass


def create_maven_project(group_id: str, artifact_id: str, workspace: Path):
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
    
BASE_POM_XML = """<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>{group_id}</groupId>
  <artifactId>workspace</artifactId>
  <version>1.0</version>
  <packaging>pom</packaging>

  <name>Workspace</name>
  <modules>
    <!-- Module werden hier hinzugefügt -->
  </modules>

  <properties>
    <maven.compiler.source>11</maven.compiler.source>
    <maven.compiler.target>11</maven.compiler.target>
  </properties>

</project>
"""


def create_parent_project(workspace: Path, group_id: str):
    """
    Create the parent project structure in the given directory.
    This assumes that the directory has been created and is empty!
    """
    pom_xml_path = workspace / "pom.xml"
    pom_xml_path.write_text(BASE_POM_XML.format(group_id=group_id, artifact_id="workspace"))


def is_workspace_structure_correct(workspace: Path, group_id: str, artifact_id: str) -> bool:
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
        fixed_state = {} # State to overwrite
        if any(workspace.iterdir()) and not is_workspace_structure_correct(workspace, group_id, artifact_id):
            fixed_state = fix_strategy.fix_structure(state)
        else:
            create_parent_project(workspace, group_id)

        # Create the transformation module (Maven project) inside the workspace
        create_maven_project(
            group_id=group_id, artifact_id=artifact_id, workspace=workspace
        )

        package_path = (
            workspace
            / artifact_id
            / "src"
            / "main"
            / "java"
            / group_id.replace(".", "/")
            / artifact_id
        )
        if not package_path.exists():
            package_path.mkdir(parents=True)

        # Create the TRANSFORMATION.md file
        transformation_md_path = workspace / artifact_id / "TRANSFORMATION.md"
        tp_parser = FileTransformationPlanParser(transformation_md_path)
        tp = TransformationPlan.parse(parser=tp_parser)
        if not transformation_md_path.exists():
            tp.update_iteration(0)

        # Create the transformation Java file (bxtool)
        bxtool_path = package_path / "BxAgentJavaBxTool.java"
        if not bxtool_path.exists():
            bxtool_path.touch()

        # Delete the App.java file created by the Maven archetype
        app_java_path = package_path / "App.java"
        if app_java_path.exists():
            app_java_path.unlink()

        # Add EMF dependencies to the pom.xml of the transformation module  
        pom_path = workspace / artifact_id / "pom.xml"
        pom_utils.add_dependencies_to_pom(pom_path, EMF_DEPENDENCIES)
        pom_utils.install_dependencies(workspace / artifact_id)

        # Update the state with the new paths and transformation plan
        new_state = PreparationState(transformation_plan=tp, bxtool_path=bxtool_path)
        new_state.update(fixed_state)
        return new_state

    return prepare_workspace_node
