from langgraph.graph import StateGraph, START, END
from mdagent.evaluation import (
    EvaluationExecutor,
)
from mdagent.agents.workflow.nodes.evaluation_node import (
    create_evaluation_node,
)

from .benchmarx import create_download_benchmarx_node
from .explore_models import create_explore_models_node
from .implementations.clear_workspace import ClearWorkspaceStrategy
from .state import PreparationState
from .prepare_workspace import create_prepare_workspace_node


def build_preparation_graph(evaluation_executor: EvaluationExecutor) -> StateGraph:
    validate_preparation_node = create_evaluation_node(
        evaluation_executor=evaluation_executor,
        mapper={
            "workspace_operability": lambda state: {
                "workspace_path": state.get("workspace_path"),
                "package_path": f"{state.get('group_id')}.{state.get('artifact_id')}",
            },
            "commands_installed": lambda state: {
                "commands": state.get("required_commands", []),
            },
        },
        execution_mode="specific",
    )
    benchmarx_node = create_download_benchmarx_node()
    explore_models_node = create_explore_models_node()
    prepare_workspace_node = create_prepare_workspace_node(fix_strategy=ClearWorkspaceStrategy())

    graph = StateGraph(PreparationState)
    graph.add_node("prepare_workspace", prepare_workspace_node)
    graph.add_node("download_benchmarx", benchmarx_node)
    graph.add_node("explore_models", explore_models_node)
    graph.add_node("validate_preparation", validate_preparation_node)

    graph.add_edge(START, "prepare_workspace")
    graph.add_edge("prepare_workspace", "download_benchmarx")
    graph.add_edge("download_benchmarx", "explore_models")
    graph.add_edge("explore_models", "validate_preparation")
    graph.add_edge("validate_preparation", END)
    return graph
