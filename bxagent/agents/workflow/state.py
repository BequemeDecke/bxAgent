from typing import TypedDict
from pathlib import Path

from bxagent.tools.audit.types import AuditRun
from bxagent.tools.transformation.plan import TransformationPlan


class WorkflowState(TypedDict):
    transformation_plan: TransformationPlan
    latest_audit_runs: list[AuditRun]
    iteration: int
    implementation_instructions: str
    written_files: list[Path]
