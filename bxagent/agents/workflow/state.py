from typing import TypedDict
from pathlib import Path

from bxagent.tools.validation.types import ValidationRun
from bxagent.tools.transformation.plan import TransformationPlan


class WorkflowState(TypedDict):
    transformation_plan: TransformationPlan
    latest_audit_runs: list[ValidationRun]
    iteration: int # Deprecated, because information is stored in the transformation plan
    implementation_instructions: str # Deprecated, because information is stored in the transformation plan
    written_files: list[Path]
