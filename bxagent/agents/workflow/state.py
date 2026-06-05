from typing import TypedDict
from pathlib import Path

from bxagent.tools.audit.types import AuditRun


class WorkflowState(TypedDict):
    transformation_source_model_description: str
    transformation_target_model_description: str
    latest_audit_runs: list[AuditRun]
    iteration: int
    implementation_instructions: str
    written_files: list[Path]
