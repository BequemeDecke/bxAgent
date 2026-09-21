from mdeagent.evaluation import EvaluationPipe, EvaluationResult, EvaluationRun
from mdeagent.evaluation.filter import (
    IsErrorFilter,
    IsExecutionRunFilter,
    IsReportCandidateFilter,
)


def filter_execution_results(
    latest_evaluation_runs: dict[str, EvaluationRun],
) -> list[EvaluationResult]:
    """
    Filter evaluation results from execution runs (JavaCompilation and FileExistence).

    Uses EvaluationPipe with IsErrorFilter to get error results and
    IsReportCandidateFilter to get report-worthy results.
    Both filters are applied separately and combined (OR logic).

    Args:
        latest_evaluation_runs: Dictionary of evaluation runs.

    Returns:
        A list of filtered evaluation results.
    """
    # Filter runs by category "execution" using IsExecutionRunFilter
    execution_pipe = EvaluationPipe() | IsExecutionRunFilter
    execution_runs = execution_pipe.filter_results(
        list(latest_evaluation_runs.values())
    )

    # Collect all results from execution runs
    all_execution_results: list[EvaluationResult] = []
    for run in execution_runs:
        all_execution_results.extend(run.results)

    if not all_execution_results:
        return []

    # Use EvaluationPipe with IsErrorFilter to get error results
    error_pipe = EvaluationPipe() | IsErrorFilter
    error_results = error_pipe.filter_results(all_execution_results)

    # Use EvaluationPipe with IsReportCandidateFilter to get report-worthy results
    report_pipe = EvaluationPipe() | IsReportCandidateFilter
    report_results = report_pipe.filter_results(all_execution_results)

    # Combine both lists, avoiding duplicates (OR logic)
    combined_results = list(
        {id(result): result for result in error_results + report_results}.values()
    )

    return combined_results