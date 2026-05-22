import asyncio
import datetime

from typing import List, Dict

from .types import Audit, AuditRun, AuditError


class AuditExecutor:
    def __init__(self, audits: List[Audit]):
        self.audits: Dict[str, Audit] = {audit.audit_id: audit for audit in audits}
        self.iterations: Dict[str, List[AuditRun]] = {
            audit.audit_id: [] for audit in audits
        }

    async def execute_all(self) -> List[AuditRun]:
        results = []
        tasks = []

        for audit in self.audits:
            started_at = datetime.datetime.now()
            tasks.append(audit.run())

        results = await asyncio.gather(*tasks)
        return results

    async def execute_specific(self, audit_id: str) -> AuditRun:
        if audit_id not in self.audits:
            raise ValueError(f"Audit with id {audit_id} not found.")

        audit = self.audits[audit_id]
        started_at = datetime.datetime.now()
        iteration = (
            self.iterations[audit_id][-1].iteration + 1
            if self.iterations[audit_id]
            else 1
        )

        try:
            run_tuple = await audit.run()
        except Exception as e:
            run_tuple = (
                [],
                [
                    AuditError(
                        message=str(e), details={"exception_type": type(e).__name__}
                    )
                ],
            )
        execution_time_ms = int(
            (datetime.datetime.now() - started_at).total_seconds() * 1000
        )

        return AuditRun(
            started_at=started_at,
            execution_time_ms=execution_time_ms,
            iteration=iteration,
            results=run_tuple[0],
            errors=run_tuple[1],
        )
