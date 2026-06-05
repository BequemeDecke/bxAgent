import asyncio
import datetime

from typing import List, Dict, TypedDict, Any
from pydantic import BaseModel

from .types import Audit, AuditRun, AuditError


class AuditInit(TypedDict):
    audit: Audit
    audit_schema: BaseModel


class AuditExecutor:
    def __init__(self, audits: Dict[str, AuditInit]):
        self.audits = audits
        self.iterations: Dict[str, List[AuditRun]] = {
            audit_id: [] for audit_id in audits
        }

    async def execute_all(self, input: Dict[str, Dict[str, Any]]) -> List[AuditRun]:
        results = []
        tasks = [
            self.execute_specific(audit_id, input=input[audit_id]) for audit_id in self.audits.keys()
        ]
        results = await asyncio.gather(*tasks)
        return results

    async def execute_specific(self, audit_id: str, input: Dict[str, Any]) -> AuditRun:
        if audit_id not in self.audits:
            raise ValueError(f"Audit with id {audit_id} not found.")

        audit_init = self.audits[audit_id]
        audit = audit_init["audit"]
        audit_schema = audit_init["audit_schema"]
         
        validated_params = audit_schema.model_validate(input)

        started_at = datetime.datetime.now()
        iteration = (
            self.iterations[audit_id][-1].iteration + 1
            if self.iterations[audit_id]
            else 1
        )

        try:
            run_tuple = await audit.run(**validated_params.model_dump())
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

        run = AuditRun(
            started_at=started_at,
            execution_time_ms=execution_time_ms,
            iteration=iteration,
            results=run_tuple[0],
            errors=run_tuple[1],
        )
        self.iterations[audit_id].append(run)
        return run

    def get_latest_results(self) -> List[AuditRun]:
        return [runs[-1] for runs in self.iterations.values() if runs]
