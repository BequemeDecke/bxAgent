from abc import ABC, abstractmethod
from typing import List, Callable

from .types import EvaluationResult


EvaluationFilter = Callable[[List[EvaluationResult]], List[EvaluationResult]]


class EvaluationPipe:
    filters: List[EvaluationFilter]

    def __init__(self):
        self.filters = []

    def filter_results(self, results: List[EvaluationResult]) -> List[EvaluationResult]:
        for filter in self.filters:
            results = filter(results)
        return results

    def add_filter(self, filter: EvaluationFilter):
        self.filters.append(filter)

    def __or__(self, other: EvaluationFilter):
        self.add_filter(other)
        return self
