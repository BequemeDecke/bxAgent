from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage

from ..state import WorkflowState


def create_call_synthesis_agent_function(synthesis_agent: CompiledStateGraph):
    def call_synthesis_agent(state: WorkflowState) -> WorkflowState:
        """
        Calls the synthesis agent with the current workflow state.

        TODO: It needs a specific schema in order to parse the output of the subagent.
        TODO: It also gets the current results of the audits, which can be used to inform the synthesis agent about what has been tried already and what the results were.
        """
        source_model = state["transformation_source_model_description"]
        target_model = state["transformation_target_model_description"]

        agent_input = (
            f"Source model description: {source_model}\n"
            f"Target model description: {target_model}\n"
            f"Results of latest audit runs: {state.get('latest_audit_runs', [])}\n"
        )

        response = synthesis_agent.invoke(
            input={"messages": [HumanMessage(content=agent_input)]}
        )

        return {"iteration": state.get("iteration", 0) + 1}

    return call_synthesis_agent
