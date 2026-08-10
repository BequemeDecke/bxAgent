from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable, TypedDict, Unpack


class EvaluationMetadata(TypedDict):
    success: bool


@dataclass
class EvaluationResult:
    content: str
    metadata: Unpack[EvaluationMetadata] = field(
        default_factory=lambda: {"success": True}
    )


@dataclass
class EvaluationError:
    message: str
    type: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationRun:
    started_at: datetime
    execution_time_ms: int
    iteration: int
    results: List[EvaluationResult]
    errors: List[EvaluationError]


StateToEvaluationMapper = Callable[[Dict[str, Any]], Dict[str, Any]]


class Evaluation(ABC):
    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def run(
        self, **kwargs
    ) -> Tuple[List[EvaluationResult], List[EvaluationError]]:
        pass
