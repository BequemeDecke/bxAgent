from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


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
