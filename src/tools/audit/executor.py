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
    
    async def execute_specific(self, audit_id: str) -> AuditRun:
        for audit in self.audits:
            if audit.audit_id == audit_id:
                return await audit.run()
        raise ValueError(f"Audit with id {audit_id} not found.")
