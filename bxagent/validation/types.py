from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable


@dataclass
class ValidationResult:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationError:
    message: str
    type: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationRun:
    started_at: datetime
    execution_time_ms: int
    iteration: int
    results: List[ValidationResult]
    errors: List[ValidationError]


StateToValidationMapper = Callable[[Dict[str, Any]], Dict[str, Any]]


class Validation(ABC):
    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def run(
        self, **kwargs
    ) -> Tuple[List[ValidationResult], List[ValidationError]]:
        pass
