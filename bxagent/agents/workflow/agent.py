from langgraph.graph import StateGraph, START, END

from bxagent.models import build_base_model
from bxagent.mapping import (
    map_workflow_to_file,
    map_workflow_to_workspace,
    map_workflow_to_commands,
)
from bxagent.agents.comprehension import build_comprehension_agent
from bxagent.validation import ValidationExecutor, implementations

from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)
from .nodes.comprehension_node import create_call_comprehension_agent_function
from .nodes.validation_node import create_validation_agent_work_function


def build_workflow_agent() -> StateGraph[WorkflowState]:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_comprehension_agent = create_call_comprehension_agent_function(
        comprehension_agent=build_comprehension_agent()
    )
    validation_executor = ValidationExecutor(
        validations={
            "workspace_operability": {
                "validation": implementations.WorkspaceOperabilityValidation(),
                "validation_schema": implementations.WorkspaceOperabilityValidationConfig,
            },
            "commands_installed": {
                "validation": implementations.CommandInstalledValidation(),
                "validation_schema": implementations.CommandInstalledValidationConfig,
            },
            "file_existence": {
                "validation": implementations.FileExistenceValidation(),
                "validation_schema": implementations.FileExistenceValidationConfig,
            },
            "java_compilation": {
                "validation": implementations.JavaCompilationValidation(),
                "validation_schema": implementations.JavaCompilationValidationConfig,
            },
        }
    )
    validation_agent_work = create_validation_agent_work_function(
        validation_executor=validation_executor,
        mapper={
            "file_existence": map_workflow_to_file,
            "java_compilation": map_workflow_to_file,
            "commands_installed": map_workflow_to_commands,
            "workspace_operability": map_workflow_to_workspace,
        },
    )

    builder = StateGraph(WorkflowState)

    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_node("call_comprehension_agent", call_comprehension_agent)
    builder.add_node("validation_agent", validation_agent_work)

    builder.add_edge(START, "call_comprehension_agent")
    builder.add_edge("call_comprehension_agent", "validation_agent")

    builder.add_conditional_edges(
        "check_transformation_iteration",
        check_transformation_iteration,
        {
            "stop": END,
            "continue": "call_comprehension_agent",
            "error": END,
        },
    )

    return builder
