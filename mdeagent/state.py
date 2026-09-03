from pathlib import Path
from typing import TypedDict

from mdeagent.comprehension.plan import TransformationPlan
from mdeagent.evaluation.types import EvaluationRun


class WorkflowState(TypedDict):
    """The State of the top level workflow graph. This state contains information and data that is relevant and needed for the entire workflow to function."""

    transformation_plan: TransformationPlan | None
    latest_evaluation_runs: list[EvaluationRun]
    written_files: list[Path]
    required_commands: list[str]
    workspace_path: Path
    transformation_package_path: str  # deprecated
    group_id: str
    artifact_id: str
    bxtool_path: Path | None

    source_model_path: Path | None
    target_model_path: Path | None
