from bxagent.tools.audit import AuditExecutor

from ..state import WorkflowState


def create_audit_agent_work_function(audit_executor: AuditExecutor):
    """
    The audits also have some states. E.g. in the first iteration, there are no files created, but the second one already has some files created, so the file existence audit should return different results. The audit agent work function needs to take care of this state and update it after each run.
    """

    async def audit_agent_work(state: WorkflowState) -> WorkflowState:
        """
        Calls the auditing core which will execute all audit implementations and update the state with the latest results.
        """
        latest_results = await audit_executor.execute_all()

        return {"latest_audit_runs": latest_results}

    return audit_agent_work
