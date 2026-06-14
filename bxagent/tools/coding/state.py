from typing import Dict, TypedDict, List
from pathlib import Path

from bxagent.tools.audit.types import AuditRun
from bxagent.tools.transformation.plan import TransformationPlan


class CodingAgentState(TypedDict):
    transformation_md: TransformationPlan
    task_specification: str  # This field will be used by a higher component
    written_java_files: List[Path]  # All of these files have to be compiled together
    transformation_implementation: str  # This field will be used by a higher component
    latest_audit_results: Dict[str, AuditRun]  # Store the results of the latest audits
