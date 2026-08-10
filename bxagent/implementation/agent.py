from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bxagent.agents.workflow.nodes.validation_node import create_validation_node
from bxagent.comprehension.plan import (
    FileTransformationPlanParser,
    TransformationPlan,
)
from bxagent.mapping import map_coding_to_file
from bxagent.models import build_base_model
from bxagent.evaluation.executor import ValidationExecutor

from .evaluate_transformation_implementation import (
    create_evaluate_transformation_implementation,
)
from .implement_bx_tool import create_implement_bx_tool_node
from .implement_transformation import create_implement_transformation_node
from .state import ImplementationState


def build_implementation_graph(
    validation_executor: ValidationExecutor,
    coding_deep_agent: CompiledStateGraph,
    workspace_path: Path,
) -> StateGraph:

    validation_executor.register_linked_validation(
        "integration_compilation", "java_compilation"
    )  # Register a module specific validation based on the java compilation implementation
    base_model = build_base_model()

    # Create implementations
    implement_transformation = create_implement_transformation_node(
        coding_agent=coding_deep_agent,
        optional_plan_factory=lambda: (
            TransformationPlan(  # Create a new transformation plan if none exists
                parser=FileTransformationPlanParser(),
            )
        ),
    )
    implement_bx_tool = create_implement_bx_tool_node(
        llm=base_model,
        workspace=workspace_path,
    )
    validation_agentic_work = create_validation_node(
        validation_executor=validation_executor,
        mapper={
            "file_existence": map_coding_to_file,
            "java_compilation": map_coding_to_file,
        },
        execution_mode="specific",
    )
    evaluate_transformation_implementation = (
        create_evaluate_transformation_implementation()
    )

    # Build the state graph
    graph = StateGraph(ImplementationState)
    graph.add_node("implement_transformation", implement_transformation, initial=True)
    graph.add_node("implement_bx_tool", implement_bx_tool)
    graph.add_node("validation_agentic_work", validation_agentic_work)

    # Add edges between the nodes to define the workflow
    graph.add_edge(START, "implement_transformation")
    graph.add_edge("implement_transformation", "implement_bx_tool")
    graph.add_edge("implement_bx_tool", "validation_agentic_work")
    graph.add_conditional_edges(
        "validation_agentic_work",
        evaluate_transformation_implementation,
        {
            "implementation_error": "implement_transformation",
            "integration_error": "implement_bx_tool",
            "max_iteration_reached": END,  # TODO: Terminate the workflow with building a failure message
            "implementation_success": END,
        },
    )

    return graph
