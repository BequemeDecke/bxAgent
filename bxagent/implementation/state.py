from typing import Dict, TypedDict, List
from pathlib import Path

from mdagent.evaluation.types import EvaluationRun
from mdagent.comprehension.plan import TransformationPlan


class ImplementationState(TypedDict):
    transformation_md: TransformationPlan
    task_specification: str  # This field will be used by a higher component
    written_java_files: List[Path]  # All of these files have to be compiled together
    bxtool_path: Path  # This field will be used by a higher component
    transformation_implementation: str  # This field will be used by a higher component
    latest_evaluation_results: Dict[str, EvaluationRun]  # Store the results of the latest evaluations
    implementation_iteration: int  # Keep track of the number of implementation iterations
