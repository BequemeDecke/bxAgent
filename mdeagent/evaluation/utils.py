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


def format_evaluation_results(results: list[EvaluationResult]) -> str:
    """
    Format a list of EvaluationResult objects into a human-readable text.

    Args:
        results: List of evaluation results to format.

    Returns:
        A formatted string representation of the evaluation results.
    """
    if not results:
        return "No evaluation results available."

    formatted_lines = []
    for i, result in enumerate(results, start=1):
        success_status = (
            "SUCCESS" if result.metadata.get("success", True) else "FAILURE"
        )
        formatted_lines.append(f"{i}. [{success_status}] {result.content}")

        # Add metadata details if present
        metadata = result.metadata
        if "file" in metadata:
            formatted_lines.append(f"   File: {metadata['file']}")
        if "line" in metadata:
            line_info = f"Line: {metadata['line']}"
            if "column" in metadata:
                line_info += f", Column: {metadata['column']}"
            formatted_lines.append(f"   {line_info}")

    return "\n".join(formatted_lines)