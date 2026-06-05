from typing import Dict, Callable

from bxagent.tools.audit import AuditExecutor
from bxagent.tools.audit.types import StateToAuditMapper

from ..state import WorkflowState


def create_audit_agent_work_function(
    audit_executor: AuditExecutor, mapper: Dict[str, StateToAuditMapper]
) -> Callable[[WorkflowState], WorkflowState]:
    """
    Creates an audit agent work function that takes a workflow state and returns the updated state with the latest audit results.
    """

    async def audit_agent_work(state: WorkflowState) -> WorkflowState:
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
