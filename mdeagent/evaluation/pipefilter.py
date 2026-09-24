from typing import Any, Callable

EvaluationFilter = Callable[[Any], list]


class EvaluationPipe:
    filters: list[EvaluationFilter]

    def __init__(self):
        self.filters = []

    def filter_results[T](self, results: list[T]) -> list[T]:
        for filter in self.filters:
            results = filter(results)
        return results

    def add_filter(self, filter: EvaluationFilter):
        self.filters.append(filter)

    def __or__(self, other: EvaluationFilter):
        self.add_filter(other)
        return self
