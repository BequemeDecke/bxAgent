from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mdeagent.comprehension.agent import build_comprehension_agent
from mdeagent.comprehension.node import create_comprehension_node
from mdeagent.evaluation import (
    EvaluationExecutor,
    FileExistenceEvaluation,
    FileExistenceSchema,
    JavaCompilationEvaluation,
    JavaCompilationSchema,
    ToolInstalledEvaluation,
    ToolInstalledSchema,
    WorkspaceStructureEvaluation,
    WorkspaceStructureSchema,
)
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.guardrails.transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from mdeagent.implementation.agent import build_implementation_graph
from mdeagent.implementation.node import create_implementation_node
from mdeagent.mapping import (
    mde_to_files,
    mde_to_maven_project,
    mde_to_tools,
    mde_to_workspace,
)
from mdeagent.preparation.agent import build_preparation_graph
from mdeagent.preparation.node import create_preparation_node
from mdeagent.state import MDEAgentState


def build_mdeagent(
    workspace_path: Path,
    benchmarx_path: Path | None = None,
    download_benchmarx: bool = False,
) -> StateGraph[MDEAgentState]:
    # 1. Initialize the core components of the MDEAgent
    check_transformation_iteration = create_check_transformation_iteration_function()
    agent_evaluator = EvaluationExecutor(
        evaluations={
            "workspace_structure": {
                "evaluation": WorkspaceStructureEvaluation(),
                "evaluation_schema": WorkspaceStructureSchema,
                "category": "preparation",
            },
            "tools_installed": {
                "evaluation": ToolInstalledEvaluation(),
                "evaluation_schema": ToolInstalledSchema,
                "category": "preparation",
            },
            "file_existence": {
                "evaluation": FileExistenceEvaluation(),
                "evaluation_schema": FileExistenceSchema,
                "category": "execution",
            },
            "java_compilation": {
                "evaluation": JavaCompilationEvaluation(),
                "evaluation_schema": JavaCompilationSchema,
                "category": "execution",
            },
        }
    )

    # 2. Create the nodes of the MDEAgent workflow
    call_comprehension_node = create_comprehension_node(
        comprehension_agent=build_comprehension_agent()
    )
    call_preparation_node = create_preparation_node(
        preparation_agent=build_preparation_graph(
            evaluation_executor=agent_evaluator,
            benchmarx_path=benchmarx_path,
            download_benchmarx=download_benchmarx,
        ).compile(),
        workspace_path=workspace_path,
        required_tools=[
            "mvn",
            "java",
            "javac",
            "jar",
        ],
    )
    call_implementation_node = create_implementation_node(
        agent=build_implementation_graph(
            evaluation_executor=agent_evaluator,
            workspace_path=workspace_path,
            benchmarx_path=benchmarx_path,
        ).compile()
    )
    call_evaluation_node = create_evaluation_node(
        evaluation_executor=agent_evaluator,
        mapper={
            "file_existence": mde_to_files,
            "java_compilation": mde_to_maven_project,
            "tools_installed": mde_to_tools,
            "workspace_structure": mde_to_workspace,
        },
    )

    # 3. Build the StateGraph for the MDEAgent workflow
    builder = StateGraph(MDEAgentState)
    builder.add_node("preparation", call_preparation_node)
    builder.add_node("comprehension", call_comprehension_node)
    builder.add_node("implementation", call_implementation_node)
    builder.add_node("evaluation", call_evaluation_node)

    builder.add_edge(START, "preparation")
    builder.add_edge("preparation", "comprehension")
    builder.add_edge("comprehension", "implementation")
    builder.add_edge("implementation", "evaluation")

    builder.add_conditional_edges(
        "evaluation",
        check_transformation_iteration,
        {
            "design_failed": "comprehension",
            "execution_failed": "implementation",
            "max_iteration_reached": END,
            "design_passed": END,
            "error": END,
        },
    )

    return builder
