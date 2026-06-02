from typing import Literal
from langchain.chat_models import BaseChatModel

from .state import WorkflowState
from bxagent.config import Config

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.WORKFLOW_APPROACH.WORKFLOW_MAX_ITERATIONS


def create_check_transformation_iteration_function(llm: BaseChatModel):
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

        # TODO: Call LLM and decide whether to continue or stop based on the audit results and the descriptions of the source and target models. For now, we will just continue.
        # It needs structured output to route the decision.

        return "stop"

    return check_transformation_iteration
