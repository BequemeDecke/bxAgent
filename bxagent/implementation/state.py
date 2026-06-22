from typing import Dict, TypedDict, List
from pathlib import Path

from bxagent.validation.types import ValidationRun
from bxagent.comprehension.plan import TransformationPlan


class ImplementationState(TypedDict):
    transformation_md: TransformationPlan
    task_specification: str  # This field will be used by a higher component
    written_java_files: List[Path]  # All of these files have to be compiled together
    bxtool_file: Path  # This field will be used by a higher component
    transformation_implementation: str  # This field will be used by a higher component
    latest_validation_results: Dict[str, ValidationRun]  # Store the results of the latest validations
    implementation_iteration: int  # Keep track of the number of implementation iterations
