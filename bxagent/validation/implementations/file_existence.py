from typing import List, Tuple
from pathlib import Path
from pydantic import BaseModel

from ..types import Validation, ValidationResult, ValidationError


class FileExistenceValidationConfig(BaseModel):
    files: List[Path]


class FileExistenceValidation(Validation):
    async def setup(self):
        pass

    async def run(self, **kwargs) -> Tuple[List[ValidationResult], List[ValidationError]]:
        config = FileExistenceValidationConfig(**kwargs)
        files = config.files
        
        results = []
        errors = []
        for file in files:
            if file.exists():
                results.append(ValidationResult(content=f"File exists: {file}"))
            else:
                errors.append(
                    ValidationError(
                        message=f"File does not exist: {file}",
                        details={"file": str(file)},
                    )
                )
        return results, errors
