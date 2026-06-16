from langchain.tools import ToolRuntime, tool
from typing import Optional

from bxagent.comprehension import TransformationPlan


@tool
def update_model_implementation(
    runtime: ToolRuntime,
    source_model_implementation: Optional[str] = None,
    target_model_implementation: Optional[str] = None,
):
    """Update the implementation details of the source and target models in the transformation plan. You can update either one or both implementations.

    Args:
        source_model_implementation (Optional[str], optional): The implementation details of the source model.
        target_model_implementation (Optional[str], optional): The implementation details of the target model.
    """
    tp: TransformationPlan = runtime.state.get("transformation_plan")
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    if (
        source_model_implementation is not None
        and target_model_implementation is not None
    ):
        tp.update_model_implementation(
            source_model_implementation, target_model_implementation
        )

    if source_model_implementation is not None and target_model_implementation is None:
        tp.update_model_implementation(
            source_model_implementation, tp.data["target_model_implementation"]
        )

    if source_model_implementation is None and target_model_implementation is not None:
        tp.update_model_implementation(
            tp.data["source_model_implementation"], target_model_implementation
        )


@tool
def update_transformation_direction(
    runtime: ToolRuntime, transformation_direction: str
):
    """Update the transformation direction in the transformation plan.

    Args:
        transformation_direction (str): The transformation direction to be updated in the transformation plan.
    """
    tp: TransformationPlan = runtime.state.get("transformation_plan")
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_transformation_direction(transformation_direction)


@tool
def update_difficulties(runtime: ToolRuntime, difficulties: str):
    """Update the identified difficulties in the transformation plan.

    Args:
        difficulties (str): The identified difficulties to be updated in the transformation plan.
    """
    tp: TransformationPlan = runtime.state.get("transformation_plan")
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_transformation_difficulties(difficulties)


@tool
def update_implementation_steps(runtime: ToolRuntime, implementation_steps: str):
    """Update the implementation steps in the transformation plan.

    Args:
        implementation_steps (str): The implementation steps in markdown to be updated in the transformation plan.
    """
    tp: TransformationPlan = runtime.state.get("transformation_plan")
    if tp is None:
        raise ValueError("Transformation plan not found in the runtime state.")

    tp.update_implementation_steps(implementation_steps)


transformation_plan_tools = [
    update_model_implementation,
    update_transformation_direction,
    update_difficulties,
    update_implementation_steps,
]
