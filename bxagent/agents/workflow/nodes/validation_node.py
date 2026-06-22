from typing import Any, Dict, Callable, Literal

from bxagent.validation import ValidationExecutor
from bxagent.validation.types import StateToValidationMapper


def create_validation_node(
    validation_executor: ValidationExecutor,
    mapper: Dict[str, StateToValidationMapper],
    execution_mode: Literal["all", "specific"] = "all",
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates an validation agent work function that takes a workflow state and returns the updated state with the latest validation results.
    """
    if execution_mode == "all":

        async def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calls the validation core which will execute all validation implementations and update the state with the latest results.
            """
            # Map the workflow state to validation parameters
            input_parameters = {}
            for validation_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[validation_name] = mapped_parameters

            # Execute all validations with the mapped parameters
            latest_results = await validation_executor.execute_all(
                input=input_parameters
            )

            return {"latest_validation_runs": latest_results}

        return validation_node

    elif execution_mode == "specific":

        async def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calls the validation core which will execute specific validation implementations based on the state and update the state with the latest results.
            """
            # Map the workflow state to validation parameters
            input_parameters = {}
            for validation_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[validation_name] = mapped_parameters

            # Execute only the validations related to transformation implementation with the mapped parameters
            latest_results = {}
            for validation_name in mapper.keys():
                validation_run = await validation_executor.execute_specific(
                    validation_id=validation_name,
                    input=input_parameters[validation_name],
                )
                latest_results[validation_name] = validation_run

            return {"latest_validation_runs": latest_results}

        return validation_node

    else:
        raise NotImplementedError(
            f"Execution mode {execution_mode} is not implemented for the validation agent work function."
        )
