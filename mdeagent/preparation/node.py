from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput

from mdeagent.preparation.state import ModelImplementation, PreparationState

from ..state import WorkflowState


def create_preparation_node(agent: CompiledStateGraph):

    async def preparation_node(state: WorkflowState) -> WorkflowState:
        workspace_path = state.get("workspace_path")
        if workspace_path is None:
            raise ValueError("Workspace path is required for the preparation agent.")

        source_model_path = state.get("source_model_path")
        if source_model_path is None:
            raise ValueError("Source model path is required for the preparation agent.")

        target_model_path = state.get("target_model_path")
        if target_model_path is None:
            raise ValueError("Target model path is required for the preparation agent.")

        group_id = state.get("group_id")
        if group_id is None:
            raise ValueError("Group ID is required for the preparation agent.")
        
        artifact_id = state.get("artifact_id")
        if artifact_id is None:
            raise ValueError("Artifact ID is required for the preparation agent.")

        prep_invoke_state = PreparationState(
            workspace_path=workspace_path,
            source_model=ModelImplementation(
                name=source_model_path.stem,  # TODO
                path=source_model_path,
                implementation=None,  # This will be set by the explore_models node after reading the model package
            ),
            target_model=ModelImplementation(
                name=target_model_path.stem,
                path=target_model_path,
                implementation=None,  # This will be set by the explore_models node after reading the model package
            ),
            group_id=group_id,
            artifact_id=artifact_id,
            required_commands=state.get("required_commands", []),
        )
        response: GraphOutput = await agent.ainvoke(prep_invoke_state, version="v2")
        prep_output_state: PreparationState = response.value

        transformation_plan = prep_output_state.get("transformation_plan")
        if transformation_plan is None:
            raise ValueError(
                "The preparation agent did not return a transformation plan, but it is required for the workflow to continue."
            )

        transformation_plan.update_model_implementation(
            source_model_implementation=prep_output_state["source_model"]["implementation"],
            target_model_implementation=prep_output_state["target_model"]["implementation"],
        )

        transformation_plan.update_package_information(
            source_model_package=prep_output_state["source_model"]["path"].stem,
            target_model_package=prep_output_state["target_model"]["path"].stem,
        )

        return {
            "transformation_plan": prep_output_state.get("transformation_plan"),
            "bxtool_path": prep_output_state.get("bxtool_path")
        }

    return preparation_node
