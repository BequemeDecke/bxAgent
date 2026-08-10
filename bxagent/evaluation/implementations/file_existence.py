from typing import List, Tuple
from pathlib import Path
from pydantic import BaseModel

from ..types import Validation, ValidationResult, ValidationError


class FileExistenceValidationConfig(BaseModel):
    files: List[Path]


class FileExistenceValidation(Validation):
    async def setup(self):
        pass

    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        config = FileExistenceValidationConfig(**kwargs)
        files = config.files

        results = []
        errors = []
        for file in files:
            if file.exists():
                results.append(
                    ValidationResult(
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
                    ValidationResult(
                        content=f"File does not exist: {file}",
                        metadata={
                            "file": str(file),
                            "success": False,
                            "include_in_report": True,
                        },
                    )
                )
        return results, errors
