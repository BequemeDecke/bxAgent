from langgraph.graph import StateGraph
from bxagent.validation import (
    ValidationExecutor,
    StateToValidationMapper,
    implementations,
)
from bxagent.agents.workflow.nodes.validation_node import (
    create_validation_agent_work_function,
)


def build_preparation_agent(validation_executor: ValidationExecutor) -> StateGraph:
    # Create the validation node for preparation
    validate_preparation_node = create_validation_agent_work_function(
        validation_executor=validation_executor,
        mapper={
            "workspace_operability": lambda state: {
                "workspace_path": state.get("workspace_path"),
                "package_path": state.get("package_path"),
            },
            "commands_installed": lambda state: {
                "commands": state.get("required_commands", []),
            },
        },
        execution_mode="specific",
    )

    graph = StateGraph()
    graph.add_node("validate_preparation", validate_preparation_node)
    return graph
