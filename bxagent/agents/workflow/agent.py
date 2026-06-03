from langgraph.graph import StateGraph, END

from bxagent.models import build_base_model
from bxagent.agents.synthesis import build_synthesis_agent

from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from .nodes.synthesis_node import create_call_synthesis_agent_function


def build_workflow_agent() -> StateGraph:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_synthesis_agent = create_call_synthesis_agent_function(
        synthesis_agent=build_synthesis_agent()
    )

    builder = StateGraph(WorkflowState)

    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_node("call_synthesis_agent", call_synthesis_agent)

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
