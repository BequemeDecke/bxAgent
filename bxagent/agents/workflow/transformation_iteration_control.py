from typing import Literal
from langchain.chat_models import BaseChatModel
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .state import WorkflowState
from bxagent.config import Config

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.WORKFLOW_APPROACH.WORKFLOW_MAX_ITERATIONS


class IterationRoute(BaseModel):
    decision: Literal["stop", "continue"] = Field(
        None,
        description="The decision on whether to stop or continue the transformation process",
    )


def create_check_transformation_iteration_function(llm: BaseChatModel):
    router = llm.with_structured_output(
        IterationRoute
    )  # This will help to check the action after the llm call

    def check_transformation_iteration(
        state: WorkflowState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> Literal["stop", "continue", "error"]:
        """
        Gate function to check if the transformation needs another iteration or not.
        """
        if state["iteration"] >= max_iterations:
            return "stop"

        runs = state["latest_audit_runs"]
        all_results = []

        for run in runs:
            if (
                len(run.errors) > 0
            ):  # If there are any errors in the audit runs, continue with the transformation process, as it might be able to fix the issues in the next iteration.
                return "continue"

            all_results.extend(run.results)

        llm_input = (
            f"Source model description: {state['transformation_source_model_description']}\n"
            f"Target model description: {state['transformation_target_model_description']}\n"
            f"Audit results from the latest iteration: {[result.content for result in all_results]}\n"
        )

        response: IterationRoute = router.invoke(
            [
                SystemMessage(
                    content="Route the input to 'continue' or 'stop' based on the audit results and the descriptions of the source and target models."
                ),
                HumanMessage(content=llm_input),
            ]
        )

        return response.decision

    return check_transformation_iteration
