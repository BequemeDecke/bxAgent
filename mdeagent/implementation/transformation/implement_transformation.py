from mdeagent.evaluation.utils import (
    filter_execution_results,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import (
    TransformationClass,
    TransformationClassGenerator,
)


def create_implement_transformation_node(
    generator: TransformationClassGenerator,
):
    """
    Creates the implement_transformation node for the implementation graph.

    This node calls the generator to synthesize the transformation class,
    then updates the transformation_class["code"] field with the generated code
    (so it can be used by implement_bx_tool if BenchmarX integration is enabled).
    """

    async def implement_transformation(
        state: ImplementationState,
    ) -> ImplementationState:
        transformation_class = state.get("transformation_class")
        if transformation_class is None:
            raise ValueError("Transformation class is not set in the state.")
        transformation_plan = state.get("transformation_plan")
        if transformation_plan is None:
            raise ValueError("Transformation plan is not set in the state.")
        latest_evaluation_runs = state.get("latest_evaluation_runs", {})
        if latest_evaluation_runs is None:
            raise ValueError("Latest evaluation runs are not set in the state.")
        task_specification = state.get("task_specification")
        if task_specification is None:
            raise ValueError("Task specification is not set in the state.")

        # Filter the evaluation results to only include those that are relevant for the current transformation class
        filtered_results = filter_execution_results(latest_evaluation_runs)

        # Call the transformation class generator to synthesize the transformation class
        written_files = await generator.synthesize_transformation_class(
            transformation_plan=transformation_plan,
            transformation_class=transformation_class,
            specific_task=task_specification,
            evaluation_results=filtered_results,
        )

        # Update the transformation_class["code"] field with the generated code
        # Read code from the first written .java file (the generated transformation class)
        code = None
        for written_file in written_files:
            if written_file.suffix == ".java":
                code = written_file.read_text(encoding="utf-8")
                break
        
        # If no code was read from written files, try reading from transformation_class path
        if code is None and transformation_class.get("path"):
            code = transformation_class["path"].read_text(encoding="utf-8")
        
        # Create updated transformation_class with the code
        updated_tc: TransformationClass = {
            "name": transformation_class.get("name", ""),
            "package": transformation_class.get("package", ""),
            "path": transformation_class.get("path"),
            "code": code,
        }
        
        # Merge old and new written files
        updated_written_files = list(set(state.get("written_files", [])) | set(written_files))

        return ImplementationState(
            transformation_plan=state["transformation_plan"],
            transformation_class=updated_tc,
            task_specification=state["task_specification"],
            maven_project_path=state["maven_project_path"],
            bxtool_path=state["bxtool_path"],
            written_files=updated_written_files,
            latest_evaluation_runs=state.get("latest_evaluation_runs", {}),
            iteration=state.get("iteration", 0),
        )

    return implement_transformation
