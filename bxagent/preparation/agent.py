from langgraph.graph import StateGraph
from bxagent.validation import (
    ValidationExecutor,
)
from bxagent.agents.workflow.nodes.validation_node import (
    create_validation_agent_work_function,
)

from .prepare_workspace import create_prepare_workspace_node


def build_preparation_agent(validation_executor: ValidationExecutor) -> StateGraph:
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
    prepare_workspace_node = create_prepare_workspace_node()

    graph = StateGraph()
    graph.add_node("prepare_workspace", prepare_workspace_node)
    graph.add_node("validate_preparation", validate_preparation_node)
    
    graph.add_edge("prepare_workspace", "validate_preparation")
    return graph
