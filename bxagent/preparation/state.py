from typing import Dict, TypedDict
from pathlib import Path

from bxagent.validation.types import ValidationRun


class PreparationState(TypedDict):
    required_commands: list[str]
    workspace_path: Path
    package_path: str
    latest_validation_results: Dict[str, ValidationRun]
