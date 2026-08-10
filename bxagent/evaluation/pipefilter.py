from abc import ABC, abstractmethod
from typing import List, Callable

from .types import ValidationResult


ValidationFilter = Callable[[List[ValidationResult]], List[ValidationResult]]


class ValidationPipe:
    filters: List[ValidationFilter]

    def __init__(self):
        self.filters = []

    def filter_results(self, results: List[ValidationResult]) -> List[ValidationResult]:
        for filter in self.filters:
            results = filter(results)
        return results

    def add_filter(self, filter: ValidationFilter):
        self.filters.append(filter)

    def __or__(self, other: ValidationFilter):
        self.add_filter(other)
        return self
