from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.state import MDEAgentState


def control_iteration(state: MDEAgentState) -> MDEAgentState:
    """
    Increments the iteration count in the state and updates it in the transformation plan.
    """
    current_iteration = state.get("iteration", 0)
    new_iteration = current_iteration + 1

    state_delta = {"iteration": new_iteration}

    serialized_plan = state.get("transformation_plan")
    if serialized_plan is not None:
        transformation_plan = TransformationPlan.from_dict(serialized_plan)
        transformation_plan.update_iteration(new_iteration)
        state_delta["transformation_plan"] = transformation_plan.to_dict()

    return MDEAgentState(**{**state, **state_delta})
