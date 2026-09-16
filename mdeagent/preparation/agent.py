from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mdeagent.evaluation import (
    EvaluationExecutor,
)
from mdeagent.evaluation.node import (
    create_evaluation_node,
)
from mdeagent.preparation.evaluate_preparation import create_evaluate_preparation
from mdeagent.preparation.explore_models import create_explore_models_node
from mdeagent.preparation.implementations.clear_workspace import ClearWorkspaceStrategy
from mdeagent.preparation.prepare_workspace import create_prepare_workspace_node
from mdeagent.preparation.state import PreparationState


def build_preparation_graph(
    evaluation_executor: EvaluationExecutor, benchmarx_path: Path | None = None, download_benchmarx: bool = False
) -> StateGraph:
    """Builder for the preparation Subgraph of the MDEAgent. This graph is responsible for preparing the workspace, exploring the source and target models, and evaluating the preparation state.

    The graph implements an agent loop: it first evaluates the current
    preparation state and then, based on a conditional edge, either (re-)prepares
    the workspace via ``prepare_workspace`` or terminates. After
    ``prepare_workspace`` the optional ``download_benchmarx`` node and the
    ``explore_models`` node are executed before the flow returns to the
    evaluation node.

    Args:
        evaluation_executor (EvaluationExecutor): The evaluation executor that will be used to evaluate the preparation state.
        benchmarx_path (Path | None, optional): The path to the benchmark data. Defaults to None.   
        download_benchmarx (bool, optional): Whether to include the download_benchmarx node in the graph. Defaults to False. This is experimental.

    Returns:
        StateGraph: The preparation subgraph of the MDEAgent.
    """
    # 1. Create the nodes of the preparation graph
    evaluate_preparation_node = create_evaluation_node(
        evaluation_executor=evaluation_executor,
        mapper={
            "workspace_structure": lambda state: {
                "workspace_path": state.get("workspace_path"),
                "artifact_id": state.get("artifact_id"),
                "package_path": f"{state.get('group_id')}.{state.get('artifact_id')}",
            },
            "tools_installed": lambda state: {
                "tools": state.get("required_tools", []),
            },
        },
        execution_mode="specific",
    )
    explore_models_node = create_explore_models_node()
    prepare_workspace_node = create_prepare_workspace_node(
        fix_strategy=ClearWorkspaceStrategy(),
        benchmarx_path=benchmarx_path,
        download_benchmarx=download_benchmarx
    )
    if download_benchmarx:
        from mdeagent.preparation.benchmarx import create_download_benchmarx_node

        benchmarx_node = create_download_benchmarx_node()

    # Conditional edge function executed after `evaluate_preparation`. It routes
    # either to `prepare_workspace` (first iteration or evaluation problems) or
    # to END (everything went smoothly / max iterations reached).
    evaluate_preparation_decision = create_evaluate_preparation()

    # 2. Build the preparation graph
    graph = StateGraph(PreparationState)
    graph.add_node("evaluate_preparation", evaluate_preparation_node)
    graph.add_node("prepare_workspace", prepare_workspace_node)
    if download_benchmarx:
        graph.add_node("download_benchmarx", benchmarx_node)
    graph.add_node("explore_models", explore_models_node)

    # The evaluation runs first. Based on its result the conditional edge either
    # starts (or restarts) the workspace preparation or finishes the subgraph.
    graph.add_edge(START, "evaluate_preparation")
    graph.add_conditional_edges(
        "evaluate_preparation",
        evaluate_preparation_decision,
        {
            "prepare_workspace": "prepare_workspace",
            "end": END,
        },
    )

    # After preparing the workspace the optional benchmarx download and the model
    # exploration are executed before the flow returns to the evaluation node.
    if download_benchmarx:
        graph.add_edge("prepare_workspace", "download_benchmarx")
        graph.add_edge("download_benchmarx", "explore_models")
    else:
        graph.add_edge("prepare_workspace", "explore_models")
    graph.add_edge("explore_models", "evaluate_preparation")

    return graph
