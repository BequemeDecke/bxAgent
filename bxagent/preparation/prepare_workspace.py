from bxagent.comprehension import TransformationPlan, FileTransformationPlanParser

from .state import PreparationState


def create_prepare_workspace_node():
    def prepare_workspace_node(state: PreparationState) -> PreparationState:
        workspace = state.get("workspace_path")
        if workspace is None:
            raise ValueError("Workspace path is not set in the state.")

        # Create workspace directory if it does not exist
        if not workspace.exists():
            workspace.mkdir(parents=True)

        # Create the src folder
        src_folder = workspace / "src"
        if not src_folder.exists():
            src_folder.mkdir()

        # Create the TRANSFORMATION.md file
        transformation_md_path = workspace / "TRANSFORMATION.md"
        tp_parser = FileTransformationPlanParser(transformation_md_path)
        tp = TransformationPlan.parse(parser=tp_parser)
        if not transformation_md_path.exists():
            transformation_md_path.touch()
            tp.update_iteration(
                0
            )  # Update the transformation plan to set the iteration to 0

        # Create the package path
        package = state.get("package_path")
        if package is not None:
            package_path = src_folder / package.replace(".", "/")
            if not package_path.exists():
                package_path.mkdir(parents=True)

        return {
            "transformation_plan": tp,
        }

    return prepare_workspace_node
