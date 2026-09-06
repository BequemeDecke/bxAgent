from langgraph.graph import END, START, StateGraph

from mdeagent.evaluation import (
    EvaluationExecutor,
)
from mdeagent.evaluation.node import (
    create_evaluation_node,
)
from mdeagent.preparation.explore_models import create_explore_models_node
from mdeagent.preparation.implementations.clear_workspace import ClearWorkspaceStrategy
from mdeagent.preparation.prepare_workspace import create_prepare_workspace_node
from mdeagent.preparation.state import PreparationState


def build_preparation_graph(
    evaluation_executor: EvaluationExecutor, download_benchmarx: bool = False
) -> StateGraph:
    """Builder for the preparation Subgraph of the MDEAgent. This graph is responsible for preparing the workspace, exploring the source and target models, and validating the preparation state.

    Args:
        evaluation_executor (EvaluationExecutor): The evaluation executor that will be used to evaluate the preparation state.
        download_benchmarx (bool, optional): Whether to include the download_benchmarx node in the graph. Defaults to False. This is experimental.

    Returns:
        StateGraph: The preparation subgraph of the MDEAgent.
    """
    # 1. Create the nodes of the preparation graph
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
    explore_models_node = create_explore_models_node()
    prepare_workspace_node = create_prepare_workspace_node(
        fix_strategy=ClearWorkspaceStrategy()
    )
    if download_benchmarx:
        from mdeagent.preparation.benchmarx import create_download_benchmarx_node

        benchmarx_node = create_download_benchmarx_node()

    # 2. Build the preparation graph
    graph = StateGraph(PreparationState)
    graph.add_node("prepare_workspace", prepare_workspace_node)
    if download_benchmarx:
        graph.add_node("download_benchmarx", benchmarx_node)
    graph.add_node("explore_models", explore_models_node)
    graph.add_node("validate_preparation", validate_preparation_node)

    graph.add_edge(START, "prepare_workspace")
    if download_benchmarx:
        graph.add_edge("prepare_workspace", "download_benchmarx")
        graph.add_edge("download_benchmarx", "explore_models")
    else:
        graph.add_edge("prepare_workspace", "explore_models")
    graph.add_edge("explore_models", "validate_preparation")
    graph.add_edge("validate_preparation", END)

    return graph
