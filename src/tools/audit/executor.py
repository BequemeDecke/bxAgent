import asyncio

from typing import List

from .types import Audit, AuditRun


class AuditExecutor:
    def __init__(self, audits: List[Audit]):
        self.audits: List[Audit] = audits

    async def execute_all(self) -> List[AuditRun]:
        results = []
        tasks = [audit.run() for audit in self.audits]
        results = await asyncio.gather(*tasks)
        return results
