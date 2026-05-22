from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TestResult:
    content: str
    format: Optional[str] = None


@dataclass
class TestError:
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestRun:
    started_at: datetime
    execution_time_ms: int
    iteration: int
    results: List[TestResult]
    errors: List[TestError]


class TestCase(ABC):
    test_id: str

    def __init__(self, test_id: str):
        self.test_id = test_id

    @abstractmethod
    async def setup(self) -> None:
        pass

    @abstractmethod
    async def run(self) -> TestRun:
        pass
