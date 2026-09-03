from typing import Callable

from .types import EvaluationResult

EvaluationFilter = Callable[[list[EvaluationResult]], list[EvaluationResult]]


class EvaluationPipe:
    filters: list[EvaluationFilter]

    def __init__(self):
        self.filters = []

    def filter_results(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        for filter in self.filters:
            results = filter(results)
        return results

    def add_filter(self, filter: EvaluationFilter):
        self.filters.append(filter)

    def __or__(self, other: EvaluationFilter):
        self.add_filter(other)
        return self
