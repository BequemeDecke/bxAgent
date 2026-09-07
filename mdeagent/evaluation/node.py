import logging
from typing import Any, Callable, Literal

from mdeagent.evaluation import EvaluationExecutor
from mdeagent.evaluation.types import StateToEvaluationMapper

logger = logging.getLogger(__name__)


def create_evaluation_node(
    evaluation_executor: EvaluationExecutor,
    mapper: dict[str, StateToEvaluationMapper],
    execution_mode: Literal["all", "specific"] = "all",
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Creates an evaluation agent work function that takes a workflow state and returns the updated state with the latest evaluation results.
    """
    if execution_mode == "all":

        async def evaluation_node(state: dict[str, Any]) -> dict[str, Any]:
            """
            Calls the evaluation core which will execute all evaluation implementations and update the state with the latest results.
            """
            logger.info("Starting evaluation node with execution mode 'all' ... (For State look at LangFuse)")
            # Map the workflow state to evaluation parameters
            input_parameters = {}
            for evaluation_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[evaluation_name] = mapped_parameters

            # Execute all evaluations with the mapped parameters
            latest_results = await evaluation_executor.execute_all(
                input=input_parameters
            )

            return {"latest_evaluation_runs": latest_results}

        return evaluation_node

    elif execution_mode == "specific":

        async def evaluation_node(state: dict[str, Any]) -> dict[str, Any]:
            """
            Calls the evaluation core which will execute specific evaluation implementations based on the state and update the state with the latest results.
            """
            logger.info("Starting evaluation node with execution mode 'specific' ... (For State look at LangFuse)")

            # Map the workflow state to evaluation parameters
            input_parameters = {}
            for evaluation_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[evaluation_name] = mapped_parameters

            # Execute only the evaluations related to transformation implementation with the mapped parameters
            latest_results = {}
            for evaluation_name in mapper.keys():
                evaluation_run = await evaluation_executor.execute_specific(
                    evaluation_id=evaluation_name,
                    input=input_parameters[evaluation_name],
                )
                latest_results[evaluation_name] = evaluation_run

            return {"latest_evaluation_runs": latest_results}

        return evaluation_node

    else:
        raise NotImplementedError(
            f"Execution mode {execution_mode} is not implemented for the evaluation agent work function."
        )
