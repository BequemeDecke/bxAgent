import logging

from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)    

PROMPT_TEMPLATE = """
--- BEGIN TRANSFORMATION PLAN ---
{transformation_plan}
--- END TRANSFORMATION PLAN ---

Use the following results to check if the transformation plan is complete and consistent:

--- BEGIN AUDIT RESULTS ---
{evaluation_results}
--- END AUDIT RESULTS ---
"""


def create_comprehension_node(comprehension_agent: CompiledStateGraph):
    async def comprehension_node(state: MDEAgentState) -> MDEAgentState:
        """
        Calls the comprehension agent with the current workflow state.

        It needs a specific schema in order to parse the output of the subagent.
        It also gets the current results of the evaluations, which can be used to inform the comprehension agent about what has been tried already and what the results were.
        """
        logger.info("Starting comprehension node ... (For State look at LangFuse)")

        serialized_transformation = state.get("transformation_plan")
        if serialized_transformation is None:
            raise ValueError(
                "The comprehension node requires a transformation plan in the state."
            )
        transformation = TransformationPlan.from_dict(serialized_transformation)

        input_prompt = PROMPT_TEMPLATE.format(
            transformation_plan=str(transformation),
            evaluation_results="\n".join(
                [str(run) for run in state.get("latest_evaluation_runs", [])]
            ),
        )

        await comprehension_agent.ainvoke(
            input={
                "messages": [HumanMessage(content=input_prompt)],
                "transformation_plan": serialized_transformation,
            },
            version="v2"
        )

        iteration = transformation.data.get("iteration", 0)
        transformation.update_iteration(iteration + 1)

        return {"transformation_plan": transformation.to_dict()}

    return comprehension_node
