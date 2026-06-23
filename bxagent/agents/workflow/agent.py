from pathlib import Path

from langgraph.graph import END, START, StateGraph

from bxagent.agents.coding.agent import build_coding_deep_agent
from bxagent.agents.comprehension import build_comprehension_agent
from bxagent.implementation import build_implementation_graph
from bxagent.mapping import (
    map_workflow_to_commands,
    map_workflow_to_file,
    map_workflow_to_workspace,
)
from bxagent.models import build_base_model
from bxagent.preparation import build_preparation_agent
from bxagent.validation import ValidationExecutor, implementations

from .nodes.comprehension_node import create_comprehension_node
from .nodes.implementation_node import create_implementation_node
from .nodes.preparation_node import create_preparation_node
from .nodes.validation_node import create_validation_node
from .state import WorkflowState
from .transformation_iteration_control import (
    create_check_transformation_iteration_function,
)


def build_workflow_agent(workspace_path: Path) -> StateGraph[WorkflowState]:
    llm = build_base_model()
    check_transformation_iteration = create_check_transformation_iteration_function(llm)
    call_comprehension_agent = create_comprehension_node(
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
    validation_agent_work = create_validation_node(
        validation_executor=validation_executor,
        mapper={
            "file_existence": map_workflow_to_file,
            "java_compilation": map_workflow_to_file,
            "commands_installed": map_workflow_to_commands,
            "workspace_operability": map_workflow_to_workspace,
        },
    )
    preparation_agent = build_preparation_agent(
        validation_executor=validation_executor
    ).compile()
    call_preparation_node = create_preparation_node(preparation_agent)
    coding_deep_agent = build_coding_deep_agent(workspace_path=workspace_path)
    implementation_agent = build_implementation_graph(
        validation_executor=validation_executor,
        coding_deep_agent=coding_deep_agent,
        workspace_path=workspace_path,
    ).compile()
    call_implementation_node = create_implementation_node(implementation_agent)

    builder = StateGraph(WorkflowState)

    # TODO: Add the rest of the workflow nodes and edges!
    builder.add_node("preparation", call_preparation_node)
    builder.add_node("comprehension", call_comprehension_agent)
    builder.add_node("implementation", call_implementation_node)
    builder.add_node("validation", validation_agent_work)

    builder.add_edge(START, "preparation")
    builder.add_edge("preparation", "comprehension")
    builder.add_edge("comprehension", "implementation")
    builder.add_edge("implementation", "validation")

    builder.add_conditional_edges(
        "validation",
        check_transformation_iteration,
        {
            "stop": END,
            "continue": "comprehension",
            "error": END,
        },
    )

    return builder
