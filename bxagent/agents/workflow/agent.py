from langgraph.graph import StateGraph, START, END

from bxagent.models import build_base_model
from bxagent.mapping import map_workflow_to_file
from bxagent.agents.synthesis import build_synthesis_agent
from bxagent.validation.implementations import (
    FileExistenceValidation,
    JavaCompilationValidation,
    FileExistenceValidationConfig,
    JavaCompilationValidationConfig,
)
from bxagent.validation.executor import ValidationExecutor

from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from .nodes.synthesis_node import create_call_synthesis_agent_function
from .nodes.validation_node import create_validation_agent_work_function


def build_workflow_agent() -> StateGraph[WorkflowState]:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_synthesis_agent = create_call_synthesis_agent_function(
        synthesis_agent=build_synthesis_agent()
    )
    validation_executor = ValidationExecutor(
        validations={
            "file_existence_validation": {
                "validation": FileExistenceValidation(),
                "validation_schema": FileExistenceValidationConfig,
            },
            "java_compilation_validation": {
                "validation": JavaCompilationValidation(),
                "validation_schema": JavaCompilationValidationConfig,
            },
        }
    )
    validation_agent_work = create_validation_agent_work_function(
        validation_executor=validation_executor,
        mapper={
            "file_existence_validation": map_workflow_to_file,
            "java_compilation_validation": map_workflow_to_file,
        },
    )

    builder = StateGraph(WorkflowState)

    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_node("call_synthesis_agent", call_synthesis_agent)
    builder.add_node("validation_agent", validation_agent_work)

    builder.add_edge(START, "call_synthesis_agent")
    builder.add_edge("call_synthesis_agent", "validation_agent")

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
