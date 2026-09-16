from pathlib import Path

from langgraph.graph import END, START, StateGraph

from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    TransformationPlan,
)
from mdeagent.evaluation.executor import EvaluationExecutor
from mdeagent.evaluation.node import create_evaluation_node
from mdeagent.implementation.evaluate_transformation_implementation import (
    create_evaluate_transformation_implementation,
)
from mdeagent.implementation.format_code import create_format_code_node
from mdeagent.implementation.implement_bx_tool import create_implement_bx_tool_node
from mdeagent.implementation.implement_transformation import (
    create_implement_transformation_node,
)
from mdeagent.implementation.state import ImplementationState
from mdeagent.mapping import implementation_to_java_files
from mdeagent.models import build_base_model


def _with_iteration_tracking(evaluation_node):
    """Wrap an evaluation node so it also advances the ``iteration`` counter.

    The implementation graph may run several work nodes per cycle
    (``implement_transformation`` and, when BenchmarX is integrated,
    ``implement_bx_tool``). The ``integration_error`` branch even routes back
    to ``implement_bx_tool`` without re-running ``implement_transformation``.
    Incrementing ``iteration`` in a work node would therefore either
    double-count (both work nodes increment) or skip the increment entirely
    (an ``integration_error`` retry that bypasses ``implement_transformation``),
    defeating the loop's safety guard.

    Advancing the counter in the evaluation node instead guarantees exactly one
    increment per graph cycle, because ``evaluate_implementation`` runs exactly
    once at the end of every cycle regardless of which work node started it.
    """

    async def evaluate_implementation(state: ImplementationState) -> dict:
        result = await evaluation_node(state)
        result["iteration"] = state.get("iteration", 0) + 1
        return result

    return evaluate_implementation


def build_implementation_graph(
    evaluation_executor: EvaluationExecutor,
    workspace_path: Path,
    benchmarx_path: Path | None = None,
) -> StateGraph:

    evaluation_executor.register_linked_evaluation(
        "integration_compilation", "java_compilation"
    )  # Register a module specific evaluation based on the java compilation implementation
    base_model = build_base_model()

    # Create implementations
    implement_transformation = create_implement_transformation_node(
        llm=base_model,
        optional_plan_factory=lambda: (
            TransformationPlan(  # Create a new transformation plan if none exists
                parser=FileTransformationPlanParser(),
            )
        ),
    )
    # Implement BxTool adapter only when BenchmarX is being used (benchmarx_path=None)
    if benchmarx_path:
        implement_bx_tool = create_implement_bx_tool_node(
            llm=base_model,
            workspace=workspace_path,
            benchmarx_path=benchmarx_path,
        )
    format_code = create_format_code_node(
        workspace=workspace_path,
    )
    evaluate_implementation_base = create_evaluation_node(
        evaluation_executor=evaluation_executor,
        mapper={
            "file_existence": implementation_to_java_files,
            "java_compilation": implementation_to_java_files,
        },
        execution_mode="specific",
    )
    # The ``iteration`` counter is advanced by the evaluation node (wrapped
    # below) and *not* by the work nodes, so it is incremented exactly once per
    # graph cycle even when the bx tool integration is active (two work nodes)
    # or when an ``integration_error`` routes back to ``implement_bx_tool``
    # without re-running ``implement_transformation``. See
    # :func:`_with_iteration_tracking` for the rationale.
    evaluate_implementation = _with_iteration_tracking(evaluate_implementation_base)
    # Conditional edge function executed after `evaluate_implementation`. It
    # routes either back to `implement_transformation` (implementation_error),
    # back to `implement_bx_tool` (integration_error, only possible when the bx
    # tool integration is part of the graph) or to END (success / max
    # iterations reached).
    evaluate_transformation_implementation = (
        create_evaluate_transformation_implementation(
            integration_enabled=bool(benchmarx_path)
        )
    )

    # Build the state graph
    graph = StateGraph(ImplementationState)
    graph.add_node("implement_transformation", implement_transformation, initial=True)
    graph.add_node("format_code", format_code)
    graph.add_node("evaluate_implementation", evaluate_implementation)

    # Add edges between the nodes to define the workflow
    graph.add_edge(START, "implement_transformation")
    graph.add_edge("format_code", "evaluate_implementation")


    # BxTool adapter flow only when BenchmarX is being used
    if benchmarx_path:
        graph.add_node("implement_bx_tool", implement_bx_tool)
        graph.add_edge("implement_transformation", "implement_bx_tool")
        graph.add_edge("implement_bx_tool", "format_code")
        graph.add_conditional_edges(
            "evaluate_implementation",
            evaluate_transformation_implementation,
            {
                "implementation_error": "implement_transformation",
                "integration_error": "implement_bx_tool",
                "max_iteration_reached": END,  # TODO: Terminate the workflow with building a failure message
                "implementation_success": END,
            },
        )
    else:
        graph.add_edge("implement_transformation", "format_code")
        # `integration_error` cannot occur when the bx tool integration is not
        # part of the graph (see `create_evaluate_transformation_implementation`),
        # so it is intentionally not part of the routing map here.
        graph.add_conditional_edges(
            "evaluate_implementation",
            evaluate_transformation_implementation,
            {
                "implementation_error": "implement_transformation",
                "max_iteration_reached": END,  # TODO: Terminate the workflow with building a failure message
                "implementation_success": END,
            },
        )

    return graph
