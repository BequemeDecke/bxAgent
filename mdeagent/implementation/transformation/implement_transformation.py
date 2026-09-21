from mdeagent.evaluation.utils import (
    filter_execution_results,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.implementation.types import TransformationClassGenerator


def create_implement_transformation_node(
    generator: TransformationClassGenerator,
):
    """
    Creates the implement_transformation node for the implementation graph.
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
        written_java_files = await generator.synthesize_transformation_class(
            transformation_plan=transformation_plan,
            transformation_class=transformation_class,
            specific_task=task_specification,
            evaluation_results=filtered_results,
        )

        # NOTE: The iteration counter is *not* advanced here. Unlike the
        # preparation subgraph (which has a single work node), the
        # implementation graph may run several work nodes per cycle
        # (``implement_transformation`` + ``implement_bx_tool``), and the
        # ``integration_error`` branch even routes back to ``implement_bx_tool``
        # without re-running ``implement_transformation``. Incrementing in a
        # work node would therefore either double-count or skip the increment
        # entirely. The counter is advanced once per cycle in the
        # ``evaluate_implementation`` node instead (see ``agent.py``).
        return {
            "written_java_files": written_java_files,
        }

    return implement_transformation
