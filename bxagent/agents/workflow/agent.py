from langgraph.graph import StateGraph, START, END

from bxagent.models import build_base_model
from bxagent.agents.synthesis import build_synthesis_agent
from bxagent.tools.audit import JavaCompilationAudit, FileExistenceAudit
from bxagent.tools.audit.executor import AuditExecutor

from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from .nodes.synthesis_node import create_call_synthesis_agent_function
from .nodes.auditing_node import create_audit_agent_work_function


def build_workflow_agent() -> StateGraph:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_synthesis_agent = create_call_synthesis_agent_function(
        synthesis_agent=build_synthesis_agent()
    )
    audit_executor = AuditExecutor(
        audits={
            "file_existence_audit": FileExistenceAudit(files=[]),
            "java_compilation_audit": JavaCompilationAudit(files=[]),
        }
    )
    audit_agent_work = create_audit_agent_work_function(audit_executor=audit_executor)

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
