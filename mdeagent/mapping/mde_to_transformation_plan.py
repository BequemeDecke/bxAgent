from typing import Any

from mdeagent.state import MDEAgentState


def mde_to_transformation_plan(state: MDEAgentState) -> dict[str, Any]:
    """Maps the transformation plan from MDEAgentState to evaluation parameters.

    Extracts the ``transformation_plan`` (``SerializedTransformationPlan``)
    from the state and returns it as a single-key dict so the evaluator
    can receive it via ``EvaluationExecutor.execute_specific``.

    Returns ``None`` when the key is missing so that downstream code can
    gracefully handle "no plan yet" instead of crashing.
    """
    return {"transformation_plan": state.get("transformation_plan")}
