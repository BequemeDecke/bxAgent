from typing import TypedDict

from bxagent.tools.audit.types import AuditRun


class WorkflowState(TypedDict):
    transformation_source_model_description: str
    transformation_target_model_description: str
    latest_audit_runs: list[AuditRun]
    iteration: int
