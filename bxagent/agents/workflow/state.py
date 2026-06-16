from typing import TypedDict
from pathlib import Path

from bxagent.validation.types import ValidationRun
from bxagent.comprehension.plan import TransformationPlan


class WorkflowState(TypedDict):
    transformation_plan: TransformationPlan
    latest_validation_runs: list[ValidationRun]
    iteration: (
        int  # Deprecated, because information is stored in the transformation plan
    )
    implementation_instructions: (
        str  # Deprecated, because information is stored in the transformation plan
    )
    written_files: list[Path]
    source_model_path: Path
    target_model_path: Path
    source_model_implementation: str
    target_model_implementation: str
