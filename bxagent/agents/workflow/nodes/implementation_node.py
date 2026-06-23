from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from bxagent.implementation.state import ImplementationState

from ..state import WorkflowState

PROMPT_TEMPLATE = "I"


def create_implementation_node(agent: CompiledStateGraph):
    """Creates a function that calls the implementation agent with the necessary state and returns the updated state after the implementation agent has done its work.

    TODO: Use the task_specification for the user to provide instructions on how to implement the transformation

    Args:
        agent (CompiledStateGraph): _description_
    """

    async def implementation_node(state: WorkflowState) -> WorkflowState:
        transformation_md = state.get("transformation_plan")
        if transformation_md is None:
            raise ValueError(
                "Transformation metadata is required for the implementation agent."
            )

        bxtool_path = state.get("bxtool_path")
        if bxtool_path is None:
            raise ValueError(
                "BXTTool file path is required for the implementation agent."
            )

        prep_invoke_state = ImplementationState(
            transformation_md=transformation_md,
            task_specification="",
            bxtool_path=bxtool_path,
        )
        response: GraphOutput = await agent.ainvoke(prep_invoke_state, version="v2")
        prep_output_state: ImplementationState = response.value

        new_written_files = set(
            state.get("written_files", [])
        )  # Get existing written files from state
        new_written_files.update(
            prep_output_state.get("written_java_files", [])
        )  # Add new written

        return {"written_files": list(new_written_files)}

    return implementation_node
