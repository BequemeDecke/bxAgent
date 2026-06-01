from pathlib import Path
from typing import List, Tuple

from ..types import Audit, AuditResult, AuditError

class FileExistenceAudit(Audit):
    def __init__(self, files: List[Path]):
        self.files = files

    async def setup(self):
        pass

    async def run(self) -> Tuple[List[AuditResult], List[AuditError]]:
        pass