from langgraph.graph import StateGraph, START, END

from bxagent.agents.workflow.nodes.auditing_node import create_audit_agent_work_function
from bxagent.tools.audit.executor import AuditExecutor
from bxagent.mapping import map_coding_to_file
from bxagent.tools.transformation.plan import (
    TransformationPlan,
    FileTransformationPlanParser,
)

from .evaluate_transformation_implementation import (
    create_evaluate_transformation_implementation,
)
from .implement_bx_tool import create_implement_bx_tool_node
from .implement_transformation import create_implement_transformation_node
from .state import CodingAgentState


def build_coding_agent_subgraph(audit_executor: AuditExecutor) -> StateGraph:
    # Create implementations
    implement_transformation = create_implement_transformation_node(
        coding_agent=None,  # TODO: Pass the actual coding agent,
        optional_plan_factory=lambda: TransformationPlan(
            parser=FileTransformationPlanParser(),
        ),
    )
    implement_bx_tool = create_implement_bx_tool_node(
        llm=None,  # TODO: Pass the actual LLM
        workspace_path=None,  # TODO: Pass the actual workspace path
    )
    audit_agentic_work = create_audit_agent_work_function(
        audit_executor=audit_executor,
        mapper={
            "file_existence": map_coding_to_file,
            "java_compilation": map_coding_to_file,
        },
        execution_mode="specific",
    )
    evaluate_transformation_implementation = (
        create_evaluate_transformation_implementation(
            llm=None,  # TODO: Pass the actual LLM
        )
    )

    # Build the state graph
    graph = StateGraph(CodingAgentState)
    graph.add_node("implement_transformation", implement_transformation, initial=True)
    graph.add_node("implement_bx_tool", implement_bx_tool)
    graph.add_node("audit_agentic_work", audit_agentic_work)

    # Add edges between the nodes to define the workflow
    graph.add_edge("implement_transformation", "implement_bx_tool")
    graph.add_edge("implement_bx_tool", "audit_agentic_work")
    graph.add_conditional_edges(
        "audit_agentic_work",
        evaluate_transformation_implementation,
        {
            "implementation_error": "implement_transformation",
            "integration_error": "implement_bx_tool",
            "implementation_success": END,
        },
    )

    return graph
