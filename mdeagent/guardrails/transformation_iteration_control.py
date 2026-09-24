import logging
from typing import Literal

from pydantic import BaseModel, Field

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.config import Config
from mdeagent.evaluation import EvaluationPipe
from mdeagent.evaluation.filter import (
    IsDesignRunFilter,
    IsErrorFilter,
    IsExecutionRunFilter,
)
from mdeagent.evaluation.types import EvaluationResult
from mdeagent.state import MDEAgentState

config = Config.get_instance()
logger = logging.getLogger(__name__)

WORKFLOW_MAX_ITERATIONS = config.AGENT_CONTROL.WORKFLOW_MAX_ITERATIONS

IterationDecision = Literal[
    "design_failed",
    "execution_failed",
    "design_passed",
    "max_iteration_reached",
    "error",
]


class IterationRoute(BaseModel):
    decision: IterationDecision = Field(
        None,
        description="The decision made by the LLM regarding the next step in the transformation iteration process.",
    )


def create_check_transformation_iteration_function():
    async def check_transformation_iteration(
        state: MDEAgentState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> IterationDecision:
        """
        Gate function to check if the transformation needs another iteration or not. Iteration is protocolled in TransformationPlan
        """
        tp = state.get("transformation_plan")
        if not tp:
            logger.error("No transformation plan found in state. Cannot check iteration.")
            return "error"

        plan = TransformationPlan.from_dict(tp)
        iteration = plan.data.get("iteration", 0)
        if iteration >= max_iterations:
            return "max_iteration_reached"

        runs = state["latest_evaluation_runs"]
        # runs is now a dict[str, EvaluationRun] – iterate over values
        runs_list = list(runs.values())
        all_results: list[EvaluationResult] = []
        logger.info(
            f"Checking transformation iteration for state: {iteration}"
        )

        for run in runs_list:
            if len(run.errors) > 0:
                logger.error(f"Errors found in evaluation run: {run.errors}")
                return "error"

            all_results.extend(run.results)

        # Build error pipe to filter out all evaluation results that are errors
        error_pipe = EvaluationPipe() | IsErrorFilter

        # Execution failed if there are error regarding java compilation or file existence
        execution_pipe = EvaluationPipe() | IsExecutionRunFilter
        execution_runs = execution_pipe.filter_results(runs_list)

        execution_errors = error_pipe.filter_results(
            [result for run in execution_runs for result in run.results]
        )
        if len(execution_errors) > 0:
            logger.info("Execution errors found in evaluation runs. Execution failed.")
            return "execution_failed"

        # Design failed if there are errors regarding the workspace structure or tools installed
        design_pipe = EvaluationPipe() | IsDesignRunFilter
        design_runs = design_pipe.filter_results(runs_list)

        design_errors = error_pipe.filter_results(
            [result for run in design_runs for result in run.results]
        )
        if len(design_errors) > 0:
            logger.info("Design errors found in evaluation runs. Design failed.")
            return "design_failed"

        logger.info("No errors found in evaluation runs. Design passed.")
        return "design_passed"

    return check_transformation_iteration
