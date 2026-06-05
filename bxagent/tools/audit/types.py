from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable


@dataclass
class AuditResult:
    content: str
    format: Optional[str] = None


@dataclass
class AuditError:
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class AuditRun:
    started_at: datetime
    execution_time_ms: int
    iteration: int
    results: List[AuditResult]
    errors: List[AuditError]


StateToAuditMapper = Callable[[Dict[str, Any]], Dict[str, Any]]


class Audit(ABC):
    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def run(self, **kwargs) -> Tuple[List[AuditResult], List[AuditError]]:
        pass
