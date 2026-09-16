from pathlib import Path
from typing import TypedDict

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun


class ImplementationState(TypedDict):
    """State for the implementation node"""

    # === Transformation ===
    transformation_md: TransformationPlan

    # === Required ===
    task_specification: str  # This field will be used by a higher component
    maven_project_path: Path  # This field will be used by a higher component
    transformation_class_path: Path  # This field will be used by a higher component
    bxtool_path: Path  # This field will be used by a higher component

    # === Implementation ===
    written_java_files: list[Path]  # All of these files have to be compiled together
    transformation_implementation: str  # This field will be used by a higher component

    # === Evaluation ===
    latest_evaluation_runs: dict[
        str, EvaluationRun
    ]  # Store the results of the latest evaluations

    # === Tracking ===
    iteration: int = 0  # Keep track of the number of implementation iterations; used by the conditional edge as a safety guard
