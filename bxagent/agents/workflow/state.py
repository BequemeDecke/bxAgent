from typing import TypedDict
from pathlib import Path

from bxagent.validation.types import ValidationRun
from bxagent.comprehension.plan import TransformationPlan


class WorkflowState(TypedDict):
    """The State of the top level workflow graph. This state contains information and data that is relevant and needed for the entire workflow to function."""

    transformation_plan: TransformationPlan
    latest_validation_runs: list[ValidationRun]
    written_files: list[Path]
    required_commands: list[str]
    workspace_path: Path
    transformation_package_path: str
    
    iteration: (
        int  # Deprecated, because information is stored in the transformation plan
    )
    implementation_instructions: (
        str  # Deprecated, because information is stored in the transformation plan
    )
    source_model_path: (
        Path  # Deprecated, because information is stored in the transformation plan
    )
    target_model_path: (
        Path  # Deprecated, because information is stored in the transformation plan
    )
    source_model_implementation: (
        str  # Deprecated, because information is stored in the transformation plan
    )
    target_model_implementation: (
        str  # Deprecated, because information is stored in the transformation plan
    )
