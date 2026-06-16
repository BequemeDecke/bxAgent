from typing import List, Tuple
from pathlib import Path
from pydantic import BaseModel

from ..types import Audit, ValidationResult, AuditError


class FileExistenceAuditConfig(BaseModel):
    files: List[Path]


class FileExistenceAudit(Audit):
    async def setup(self):
        pass

    async def run(self, **kwargs) -> Tuple[List[ValidationResult], List[AuditError]]:
        config = FileExistenceAuditConfig(**kwargs)
        files = config.files
        
        results = []
        errors = []
        for file in files:
            if file.exists():
                results.append(ValidationResult(content=f"File exists: {file}"))
            else:
                errors.append(
                    AuditError(
                        message=f"File does not exist: {file}",
                        details={"file": str(file)},
                    )
                )
        return results, errors
