from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable
from bxagent.tools.audit import AuditExecutor

from ..state import WorkflowState


class StateToAuditParameterMapper(ABC):
    """
    This class is responsible for mapping the workflow state to the parameters required by the audits. This way, we can decouple the audits from the specific structure of the workflow state and make it easier to add new audits in the future without having to change the workflow state structure.
    """

    @abstractmethod
    def map(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Maps the workflow state to a dictionary of parameters that can be passed to the audits. The keys of the dictionary should match the parameter names expected by the audits.
        """
        pass


def create_audit_agent_work_function(
    audit_executor: AuditExecutor, mapper: List[StateToAuditParameterMapper]
) -> Callable[[WorkflowState], WorkflowState]:
    """
    Creates an audit agent work function that takes a workflow state and returns the updated state with the latest audit results.
    """
    

    async def audit_agent_work(state: WorkflowState) -> WorkflowState:
        """
        Calls the auditing core which will execute all audit implementations and update the state with the latest results.
        """
        # Map the workflow state to audit parameters
        audit_parameters = {}
        for m in mapper:
            audit_parameters.update(m.map(state))

        # Execute all audits with the mapped parameters
        latest_results = await audit_executor.execute_all(**audit_parameters)

        return {"latest_audit_runs": latest_results}

    return audit_agent_work
