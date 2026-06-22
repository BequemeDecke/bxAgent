from pathlib import Path
from typing import Dict, TypedDict, Optional

from bxagent.comprehension.plan import TransformationPlan
from bxagent.validation.types import ValidationRun


class PreparationState(TypedDict):
    required_commands: list[str]
    workspace_path: Path
    package_path: str
    latest_validation_runs: Dict[str, ValidationRun] = []
    source_model_path: Path
    target_model_path: Path
    source_model_implementation: str = ""
    target_model_implementation: str = ""
    transformation_plan: Optional[TransformationPlan] = None
