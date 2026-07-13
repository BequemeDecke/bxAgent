from pathlib import Path
from typing import Dict, TypedDict, Optional

from bxagent.comprehension.plan import TransformationPlan
from bxagent.validation.types import ValidationRun


class ModelImplementation(TypedDict):
    name: str  # The name of the model (e.g. Family, Person); Register and Factory classes are derived from this name (e.g. FamilyRegister, FamilyFactory)
    path: Path  # The path to the model package
    implementation: Optional[str]  # Aggregate of all Java files in the model package including their file names; This has to be set by the explore_models node after reading the model package


class PreparationState(TypedDict):
    required_commands: list[str] # List of required commands to be available in the system PATH in order to run the agent properly
    workspace_path: Path
    package_path: str # deprecated: It's now groupId and artifactId
    latest_validation_runs: Dict[str, ValidationRun] = []
    transformation_plan: Optional[TransformationPlan] = None
    bxtool_path: Optional[Path] = None
    source_model: ModelImplementation
    target_model: ModelImplementation

    group_id: Optional[str] = None  # Maven groupId for the generated project
    artifact_id: Optional[str] = None  # Maven artifactId for the generated project
    install_benchmarx: bool = True  # Whether to install the benchmarx tool in the workspace
    benchmarx_path: Optional[Path] = None  # Path to the benchmarx tool in the workspace
