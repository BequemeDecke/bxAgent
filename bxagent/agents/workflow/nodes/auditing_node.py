from typing import Any, Dict, Callable, Literal

from bxagent.tools.validation import AuditExecutor
from bxagent.tools.validation.types import StateToAuditMapper


def create_audit_agent_work_function(
    audit_executor: AuditExecutor,
    mapper: Dict[str, StateToAuditMapper],
    execution_mode: Literal["all", "specific"] = "all",
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Creates an audit agent work function that takes a workflow state and returns the updated state with the latest audit results.
    """
    if execution_mode == "all":

        async def audit_agent_work(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calls the auditing core which will execute all audit implementations and update the state with the latest results.
            """
            # Map the workflow state to audit parameters
            input_parameters = {}
            for audit_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[audit_name] = mapped_parameters

            # Execute all audits with the mapped parameters
            latest_results = await audit_executor.execute_all(input=input_parameters)

            return {"latest_audit_runs": latest_results}

        return audit_agent_work

    elif execution_mode == "specific":

        async def audit_agent_work(state: Dict[str, Any]) -> Dict[str, Any]:
            """
            Calls the auditing core which will execute specific audit implementations based on the state and update the state with the latest results.
            """
            # Map the workflow state to audit parameters
            input_parameters = {}
            for audit_name, map_state in mapper.items():
                mapped_parameters = map_state(state)
                input_parameters[audit_name] = mapped_parameters

            # Execute only the audits related to transformation implementation with the mapped parameters
            latest_results = {}
            for audit_name in mapper.keys():
                audit_run = await audit_executor.execute_specific(
                    audit_id=audit_name, input=input_parameters[audit_name]
                )
                latest_results[audit_name] = audit_run

            return {"latest_audit_runs": latest_results}

        return audit_agent_work

    else:
        raise NotImplementedError(
            f"Execution mode {execution_mode} is not implemented for the audit agent work function."
        )
