from typing import TypedDict, Optional
from pathlib import Path

from bxagent.validation.types import ValidationRun
from bxagent.comprehension.plan import TransformationPlan


class WorkflowState(TypedDict):
    """The State of the top level workflow graph. This state contains information and data that is relevant and needed for the entire workflow to function."""

    transformation_plan: Optional[TransformationPlan]
    latest_validation_runs: list[ValidationRun]
    written_files: list[Path]
    required_commands: list[str]
    workspace_path: Path
    transformation_package_path: str
