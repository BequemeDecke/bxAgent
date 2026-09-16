from mdeagent.evaluation.pipefilter import EvaluationFilter
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun


def _is_report_candidate_filter(
    results: list[EvaluationResult],
) -> list[EvaluationResult]:
    """
    Filter function to determine if evaluation results should be included in the report.
    Returns a list of results that should be included.
    """
    return [
        result for result in results if result.metadata.get("include_in_report") is True
    ]


IsReportCandidateFilter: EvaluationFilter = _is_report_candidate_filter


def _is_error_filter(
    results: list[EvaluationResult],
) -> list[EvaluationResult]:
    """
    Filter function to determine if evaluation results are errors.
    Returns a list of results that are errors.
    """
    return [result for result in results if result.metadata.get("success") is False]


IsErrorFilter: EvaluationFilter = _is_error_filter


def _is_execution_run(
    runs: list[EvaluationRun],
) -> list[EvaluationResult]:
    """
    Filter function to determine if evaluation runs contain execution errors.
    Returns a list of results that are execution errors.
    """
    return [run.category == "execution" for run in runs if len(run.results) > 0]


IsExecutionRunFilter: EvaluationFilter = _is_execution_run


def _is_design_run(
    runs: list[EvaluationRun],
) -> list[EvaluationResult]:
    """
    Filter function to determine if evaluation runs contain design errors.
    Returns a list of results that are design errors.
    """
    return [run.category == "design" for run in runs if len(run.results) > 0]

IsDesignRunFilter: EvaluationFilter = _is_design_run
