from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TypedDict, Unpack


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
    details: dict[str, Any] | None = None


@dataclass
class EvaluationRun:
    started_at: datetime
    execution_time_ms: int
    iteration: int
    results: list[EvaluationResult]
    errors: list[EvaluationError]


StateToEvaluationMapper = Callable[[dict[str, Any]], dict[str, Any]]


class Evaluation(ABC):
    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def run(
        self, **kwargs
    ) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        pass
