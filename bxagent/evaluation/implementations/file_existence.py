from typing import List, Tuple
from pathlib import Path
from pydantic import BaseModel

from ..types import Evaluation, EvaluationResult, EvaluationError


class FileExistenceEvaluationConfig(BaseModel):
    files: List[Path]


class FileExistenceEvaluation(Evaluation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[EvaluationResult], List[EvaluationError]]:
        config = FileExistenceEvaluationConfig(**kwargs)
        files = config.files

        results = []
        errors = []
        for file in files:
            if file.exists():
                results.append(
                    EvaluationResult(
                        content=f"File exists: {file}",
                        metadata={
                            "file": str(file),
                            "success": True,
                            "include_in_report": False,
                        },
                    )
                )
            else:
                results.append(
                    EvaluationResult(
                        content=f"File does not exist: {file}",
                        metadata={
                            "file": str(file),
                            "success": False,
                            "include_in_report": True,
                        },
                    )
                )
        return results, errors
