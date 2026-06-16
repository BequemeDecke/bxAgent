from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage

from bxagent.agents.comprehension import ComprehensionResponseFormat
from ..state import WorkflowState

PROMPT_TEMPLATE = """
--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

Use the following results to check if the transformation plan is complete and consistent:

--- BEGIN AUDIT RESULTS ---
{validation_results}
--- END AUDIT RESULTS ---
"""


def create_call_comprehension_agent_function(comprehension_agent: CompiledStateGraph):
    def call_comprehension_agent(state: WorkflowState) -> WorkflowState:
        """
        Calls the comprehension agent with the current workflow state.

        It needs a specific schema in order to parse the output of the subagent.
        It also gets the current results of the validations, which can be used to inform the comprehension agent about what has been tried already and what the results were.
        """
        transformation = state.get("transformation_plan")

        input_prompt = PROMPT_TEMPLATE.format(
            transformation_plan=str(transformation),
            validation_results="\n".join(
                [str(run) for run in state.get("latest_validation_runs", [])]
            ),
        )

        result = comprehension_agent.invoke(
            input={"messages": [HumanMessage(content=input_prompt)]},
        )
        structured_response: ComprehensionResponseFormat = result["structured_response"]
        return {
            "iteration": state.get("iteration", 0) + 1,
            "implementation_instructions": structured_response.implementation_instructions,
        }

    return call_comprehension_agent
