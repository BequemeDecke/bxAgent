from typing import TypedDict

from bxagent.tools.audit.types import AuditRun


class WorkflowState(TypedDict):
    transformation_source_model: str
    transformation_target_model: str
    latest_audit_runs: list[AuditRun]
