from pathlib import Path
from typing import List, Tuple

from pydantic import BaseModel

from ..types import Evaluation, EvaluationError, EvaluationResult


class WorkspaceOperabilityEvaluationConfig(BaseModel):
    workspace_path: Path
    package_path: str


class WorkspaceOperabilityEvaluation(Evaluation):
    async def setup(self) -> None:
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[EvaluationResult], List[EvaluationError]]:
        results: List[EvaluationResult] = []
        errors: List[EvaluationError] = []

        # Check if the workspace path exists and is a directory
        workspace_path: Path = kwargs.get("workspace_path")
        if not workspace_path.exists() or not workspace_path.is_dir():
            results.append(
                EvaluationResult(
                    content=f"Workspace path '{workspace_path}' does not exist or is not a directory.",
                    metadata={"success": False, "include_in_report": False},
                )
            )

        # Check if the TRANSFORMATION.md file exists in the workspace
        transformation_md_path = workspace_path / "TRANSFORMATION.md"
        if not transformation_md_path.exists() or not transformation_md_path.is_file():
            results.append(
                EvaluationResult(
                    content="Required file 'TRANSFORMATION.md' is missing in the workspace.",
                    metadata={"success": False, "include_in_report": False},
                )
            )

        # Check if the src folder exists in the workspace
        src_folder = workspace_path / "src"
        if not src_folder.exists() or not src_folder.is_dir():
            results.append(
                EvaluationResult(
                    content="Required folder 'src' is missing in the workspace.",
                    metadata={"success": False, "include_in_report": False},
                )
            )

        # Check if the package path exists within the src folder e.g. de.hof-university.bxagent
        package_path = kwargs.get("package_path")
        package_parts = package_path.split(".")
        current_path = src_folder
        for part in package_parts:
            current_path = current_path / part
            if not current_path.exists() or not current_path.is_dir():
                results.append(
                    EvaluationResult(
                        content=f"Package path '{package_path}' is invalid. Missing directory: '{current_path}'.",
                        metadata={"success": False, "include_in_report": False},
                    )
                )
                break

        return results, errors
