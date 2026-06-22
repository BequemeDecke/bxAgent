from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from bxagent.preparation.state import PreparationState

from ..state import WorkflowState


def create_call_preparation_agent_node(agent: CompiledStateGraph):

    async def call_preparation_agent(state: WorkflowState) -> WorkflowState:
        workspace_path = state.get("workspace_path")
        if workspace_path is None:
            raise ValueError("Workspace path is required for the preparation agent.")

        source_model_path = state.get("source_model_path")
        if source_model_path is None:
            raise ValueError("Source model path is required for the preparation agent.")

        target_model_path = state.get("target_model_path")
        if target_model_path is None:
            raise ValueError("Target model path is required for the preparation agent.")

        package_path = state.get("transformation_package_path")
        if package_path is None:
            raise ValueError(
                "Transformation package path is required for the preparation agent."
            )

        prep_invoke_state = PreparationState(
            workspace_path=workspace_path,
            source_model_path=source_model_path,
            target_model_path=target_model_path,
            package_path=package_path,
            required_commands=state.get("required_commands", []),
        )
        response: GraphOutput = await agent.ainvoke(prep_invoke_state, version="v2")
        prep_output_state: PreparationState = response.value

        return {
            "source_model_implementation": prep_output_state.get(
                "source_model_implementation"
            ),
            "target_model_implementation": prep_output_state.get(
                "target_model_implementation"
            ),
        }

    return call_preparation_agent
