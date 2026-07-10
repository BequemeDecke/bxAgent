import subprocess

from bxagent.comprehension import TransformationPlan, FileTransformationPlanParser

from .state import PreparationState


def create_prepare_workspace_node():
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

        # Create workspace directory if it does not exist
        if not workspace.exists():
            workspace.mkdir(parents=True)

        # Create the maven project
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

        return {
            "transformation_plan": tp,
            "bxtool_path": bxtool_path,
        }

    return prepare_workspace_node
