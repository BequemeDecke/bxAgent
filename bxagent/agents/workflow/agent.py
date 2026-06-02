from langgraph.graph import StateGraph, END

from bxagent.models import build_base_model
from .state import WorkflowState
from .transformation_iteration_control import create_check_transformation_iteration_function

def build_workflow_agent() -> StateGraph:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    
    builder = StateGraph(WorkflowState)
    
    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_conditional_edges(
        "check_transformation_iteration",
        check_transformation_iteration,
        {
            "stop": END,
            "continue": "continue_node",
            "error": END,
        },
    )
    
    return builder