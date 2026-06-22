from langgraph.graph import StateGraph, START, END
from bxagent.validation import (
    ValidationExecutor,
)
from bxagent.agents.workflow.nodes.validation_node import (
    create_validation_node,
)

from .explore_models import create_explore_models_node
from .state import PreparationState
from .prepare_workspace import create_prepare_workspace_node


def build_preparation_agent(validation_executor: ValidationExecutor) -> StateGraph:
    validate_preparation_node = create_validation_node(
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
    explore_models_node = create_explore_models_node()
    prepare_workspace_node = create_prepare_workspace_node()

    graph = StateGraph(PreparationState)
    graph.add_node("prepare_workspace", prepare_workspace_node)
    graph.add_node("explore_models", explore_models_node)
    graph.add_node("validate_preparation", validate_preparation_node)

    graph.add_edge(START, "prepare_workspace")
    graph.add_edge("prepare_workspace", "explore_models")
    graph.add_edge("explore_models", "validate_preparation")
    graph.add_edge("validate_preparation", END)
    return graph
