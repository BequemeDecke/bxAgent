from pathlib import Path
from typing import List, Tuple

from ..types import Audit, AuditResult, AuditError


class FileExistenceAudit(Audit):
    async def setup(self):
        pass

    async def run(self, **kwargs) -> Tuple[List[AuditResult], List[AuditError]]:
        if "files" not in kwargs:
            raise ValueError("Missing required parameter: 'files'")
        
        files = kwargs["files"]
        if not isinstance(files, list):
            raise ValueError("Parameter 'files' must be a list of file paths.")
        
        results = []
        errors = []
        for file in files:
            if file.exists():
                results.append(AuditResult(content=f"File exists: {file}"))
            else:
                errors.append(
                    AuditError(
                        message=f"File does not exist: {file}",
                        details={"file": str(file)},
                    )
                )
        return results, errors
