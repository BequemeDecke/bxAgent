from typing import Any, Union

from mdeagent.evaluation.pipefilter import EvaluationFilter
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun


def _extract_runs(runs: Union[dict[str, EvaluationRun], list[EvaluationRun]]) -> list[EvaluationRun]:
    """Extract list of EvaluationRun from dict or list input."""
    if isinstance(runs, dict):
        return list(runs.values())
    return runs


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
    runs: Any,
) -> list[EvaluationRun]:
    """
    Filter function to determine if evaluation runs contain execution errors.
    Accepts dict[str, EvaluationRun] or list[EvaluationRun].
    Returns a list of runs that are execution runs.
    """
    runs_list = _extract_runs(runs)
    return [run for run in runs_list if run.category == "execution" and len(run.results) > 0]


IsExecutionRunFilter: EvaluationFilter = _is_execution_run


def _is_design_run(
    runs: Any,
) -> list[EvaluationRun]:
    """
    Filter function to determine if evaluation runs contain design errors.
    Accepts dict[str, EvaluationRun] or list[EvaluationRun].
    Returns a list of runs that are design runs.
    """
    runs_list = _extract_runs(runs)
    return [run for run in runs_list if run.category == "design" and len(run.results) > 0]


IsDesignRunFilter: EvaluationFilter = _is_design_run
