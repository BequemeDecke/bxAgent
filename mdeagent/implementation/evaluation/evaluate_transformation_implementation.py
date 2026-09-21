"""Conditional edge function for the implementation graph.

This module contains the routing decision that is executed after the
``evaluate_implementation`` node. Based on the current ``iteration`` and the
latest evaluation results it decides whether the transformation has to be
(re-)implemented (``implementation_error`` -> ``implement_transformation``),
whether the bx tool integration has to be (re-)implemented
(``integration_error`` -> ``implement_bx_tool``) or whether the implementation
subgraph is done (``implementation_success`` / ``max_iteration_reached`` ->
END).

The logic mirrors :mod:`mdeagent.preparation.evaluate_preparation` so that both
subgraphs follow the same routing pattern.

Note: an ``integration_error`` can only be returned when the bx tool
integration is actually part of the graph (``integration_enabled=True``). When
no bx tool is integrated, every problem is reported as an
``implementation_error`` (routing back to ``implement_transformation``).
"""

from typing import Literal

from mdeagent.config import Config

from ..state import ImplementationState

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.AGENT_CONTROL.WORKFLOW_MAX_ITERATIONS


EvaluationDecision = Literal[
    "implementation_error",
    "integration_error",
    "implementation_success",
    "max_iteration_reached",
]


def _run_has_problems(run: object) -> bool:
    """Return ``True`` if an :class:`EvaluationRun` reported a problem.

    A run is considered problematic if it raised an :class:`EvaluationError`
    (``run.errors``) or if it produced an :class:`EvaluationResult` whose
    ``success`` metadata is explicitly ``False``. The implementation
    evaluations (e.g. :class:`FileExistenceEvaluation`) signal problems via
    results with ``success=False`` instead of raising errors, so both have to
    be checked (mirroring
    :func:`mdeagent.preparation.evaluate_preparation._evaluation_has_problems`).
    """
    if len(run.errors) > 0:  # type: ignore[attr-defined]
        return True
    return any(
        result.metadata.get("success", True) is False
        for result in run.results  # type: ignore[attr-defined]
    )


def create_evaluate_transformation_implementation(
    *, integration_enabled: bool = False
):
    """Create the conditional edge function for the implementation graph.

    Args:
        integration_enabled: Whether the ``implement_bx_tool`` node is part of
            the graph. Only when this is ``True`` can an ``integration_error``
            be returned (and routed to ``implement_bx_tool``). When ``False``
            (no bx tool integration) every problem is reported as an
            ``implementation_error`` instead, routing back to
            ``implement_transformation``.
    """

    def evaluate_transformation_implementation(
        agent_state: ImplementationState,
        max_iterations: int = WORKFLOW_MAX_ITERATIONS,
    ) -> EvaluationDecision:
        """Routing decision executed after the ``evaluate_implementation`` node.

        - ``max_iterations`` acts as a safety guard to avoid infinite loops.
        - When the bx tool integration is part of the graph, problems in the
          ``integration_compilation`` evaluation route back to
          ``implement_bx_tool`` (``integration_error``). This decision can only
          be returned when ``integration_enabled`` is ``True``.
        - Any other problem routes back to ``implement_transformation``
          (``implementation_error``).
        - If there are no problems the implementation is considered successful
          (``implementation_success`` -> END).
        """
        iteration = agent_state.get("iteration", 0)
        if iteration >= max_iterations:
            return "max_iteration_reached"

        latest_results = agent_state.get("latest_evaluation_runs", {})

        # An integration error (bx tool / benchmarx integration) can only occur
        # when the ``implement_bx_tool`` node is part of the graph. When the
        # integration is disabled, integration problems are reported as
        # implementation errors instead.
        if integration_enabled:
            integration_run = latest_results.get("integration_compilation")
            # Only treat a present, failing integration run as an integration
            # error. A missing run (e.g. evaluation not executed) is not an
            # integration error on its own.
            if integration_run is not None and _run_has_problems(integration_run):
                return "integration_error"
            # Exclude the (clean) integration run from the implementation check.
            implementation_runs = {
                key: run
                for key, run in latest_results.items()
                if key != "integration_compilation"
            }
        else:
            # No bx tool integration: every problem is an implementation error.
            implementation_runs = latest_results

        if any(_run_has_problems(run) for run in implementation_runs.values()):
            return "implementation_error"

        # No errors and no failing results -> implementation is complete.
        return "implementation_success"

    return evaluate_transformation_implementation
