from typing import List

from ..pipefilter import EvaluationFilter
from ..types import EvaluationResult


def _is_report_candidate_filter(
    results: List[EvaluationResult],
) -> List[EvaluationResult]:
    """
    Filter function to determine if evaluation results should be included in the report.
    Returns a list of results that should be included.
    """
    return [
        result for result in results if result.metadata.get("include_in_report") is True
    ]


IsReportCandidateFilter: EvaluationFilter = _is_report_candidate_filter


def _is_error_filter(
    results: List[EvaluationResult],
) -> List[EvaluationResult]:
    """
    Filter function to determine if evaluation results are errors.
    Returns a list of results that are errors.
    """
    return [result for result in results if result.metadata.get("success") is False]


IsErrorFilter: EvaluationFilter = _is_error_filter
