from typing import List

from ..pipefilter import ValidationFilter
from ..types import ValidationResult


def _is_report_candidate_filter(
    results: List[ValidationResult],
) -> List[ValidationResult]:
    """
    Filter function to determine if validation results should be included in the report.
    Returns a list of results that should be included.
    """
    return [
        result for result in results if result.metadata.get("include_in_report") is True
    ]


IsReportCandidateFilter: ValidationFilter = _is_report_candidate_filter


def _is_error_filter(
    results: List[ValidationResult],
) -> List[ValidationResult]:
    """
    Filter function to determine if validation results are errors.
    Returns a list of results that are errors.
    """
    return [result for result in results if result.metadata.get("success") is False]


IsErrorFilter: ValidationFilter = _is_error_filter
