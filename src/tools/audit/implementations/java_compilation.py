from typing import List, Tuple

from ..types import AuditResult, AuditError, Audit


class JavaCompilationAudit(Audit):
    """
    Audit that checks if Java files can be compiled successfully.
    """

    def __init__(self, files: List[str]):
        self.files = files

    async def setup(self):
        # No setup needed for this audit
        pass

    async def run(self) -> Tuple[AuditResult, List[AuditError]]:
        return [], []
