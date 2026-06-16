from langgraph.graph import StateGraph, START, END

from bxagent.models import build_base_model
from bxagent.mapping import map_workflow_to_file
from bxagent.agents.synthesis import build_synthesis_agent
from bxagent.tools.validation.implementations import (
    FileExistenceAudit,
    JavaCompilationAudit,
    FileExistenceAuditConfig,
    JavaCompilationAuditConfig,
)
from bxagent.tools.validation.executor import ValidationExecutor

from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from .nodes.synthesis_node import create_call_synthesis_agent_function
from .nodes.auditing_node import create_audit_agent_work_function


def build_workflow_agent() -> StateGraph[WorkflowState]:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_synthesis_agent = create_call_synthesis_agent_function(
        synthesis_agent=build_synthesis_agent()
    )
    audit_executor = ValidationExecutor(
        audits={
            "file_existence_audit": {
                "audit": FileExistenceAudit(),
                "audit_schema": FileExistenceAuditConfig,
            },
            "java_compilation_audit": {
                "audit": JavaCompilationAudit(),
                "audit_schema": JavaCompilationAuditConfig,
            },
        }
    )
    audit_agent_work = create_audit_agent_work_function(
        audit_executor=audit_executor,
        mapper={
            "file_existence_audit": map_workflow_to_file,
            "java_compilation_audit": map_workflow_to_file,
        },
    )

    builder = StateGraph(WorkflowState)

    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_node("call_synthesis_agent", call_synthesis_agent)
    builder.add_node("audit_agent", audit_agent_work)

    builder.add_edge(START, "call_synthesis_agent")
    builder.add_edge("call_synthesis_agent", "audit_agent")

    builder.add_conditional_edges(
        "check_transformation_iteration",
        check_transformation_iteration,
        {
            "stop": END,
            "continue": "call_synthesis_agent",
            "error": END,
        },
    )

    return builder
