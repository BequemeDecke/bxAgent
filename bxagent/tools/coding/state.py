from typing import TypedDict, List
from pathlib import Path


class CodingAgentState(TypedDict):
    transformation_md: str  # Content of the TRANSFORMATION.md file
    task_specification: str  # This field will be used by a higher component
    written_java_files: List[Path]  # All of these files have to be compiled together
    source_model_package: str  # The java package where the source model is located
    target_model_package: str  # The java package where the target model is located
