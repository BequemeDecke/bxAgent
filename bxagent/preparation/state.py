from typing import Dict, TypedDict
from pathlib import Path

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
