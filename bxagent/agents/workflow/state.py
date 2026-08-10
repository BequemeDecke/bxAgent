from typing import TypedDict, Optional
from pathlib import Path

from bxagent.evaluation.types import EvaluationRun
from bxagent.comprehension.plan import TransformationPlan


class WorkflowState(TypedDict):
    """The State of the top level workflow graph. This state contains information and data that is relevant and needed for the entire workflow to function."""

    transformation_plan: Optional[TransformationPlan]
    latest_evaluation_runs: list[EvaluationRun]
    written_files: list[Path]
    required_commands: list[str]
    workspace_path: Path
    transformation_package_path: str # deprecated
    group_id: str
    artifact_id: str
    bxtool_path: Optional[Path]

    source_model_path: Optional[Path]
    target_model_path: Optional[Path]
