"""Conditional edge function for the preparation graph.

This module contains the routing decision that is executed after the
``evaluate_preparation`` node. Based on the current ``iteration`` and the
latest evaluation results it decides whether the workspace has to be
(re-)prepared (``structure_incomplete``) or whether the preparation subgraph is
done (``workspace_prepared`` -> END).

The logic mirrors :mod:`mdeagent.implementation.evaluate_transformation_implementation`
so that both subgraphs follow the same routing pattern.
"""

from typing import Literal

from mdeagent.config import Config

from .state import PreparationState

config = Config.get_instance()

WORKFLOW_MAX_ITERATIONS = config.AGENT_CONTROL.WORKFLOW_MAX_ITERATIONS


PreparationDecision = Literal[
    "structure_incomplete",
    "workspace_prepared",
]


def _evaluation_has_problems(state: PreparationState) -> bool:
    """Return ``True`` if any of the latest evaluation runs reported an error.

    A run is considered problematic if it raised an :class:`EvaluationError`
    (``run.errors``) or if it produced an :class:`EvaluationResult` whose
    ``success`` metadata is explicitly ``False``. The preparation evaluations
    (e.g. ``WorkspaceStructureEvaluation``) signal problems via results with
    ``success=False`` instead of raising errors, so both have to be checked.
    """
    latest_results = state.get("latest_evaluation_runs", {})
    # ``execution_mode="specific"`` (used by the preparation graph) returns a
    # ``dict[str, EvaluationRun]`` while ``execution_mode="all"`` returns a
    # ``list[EvaluationRun]``. Handle both to stay robust.
    runs = (
        latest_results.values()
        if isinstance(latest_results, dict)
        else latest_results
    )

    for run in runs:
        if len(run.errors) > 0:
            return True
        if any(
            result.metadata.get("success", True) is False for result in run.results
        ):
            return True
    return False


def create_evaluate_preparation():
    def evaluate_preparation(
        state: PreparationState, max_iterations: int = WORKFLOW_MAX_ITERATIONS
    ) -> PreparationDecision:
        """Routing decision executed after the ``evaluate_preparation`` node.

        - The very first run (``iteration == 0``) always routes to
          ``structure_incomplete`` because the workspace has not been prepared yet
          and the evaluation results are therefore expected to fail.
        - On every subsequent run the latest evaluation results are inspected:
          if everything went smoothly we route to ``workspace_prepared`` (END), otherwise we
          route back to ``structure_incomplete`` to give the fix strategy another
          chance to repair the workspace.
        - ``max_iterations`` acts as a safety guard to avoid infinite loops.
        """
        iteration = state.get("iteration", 0)
        if iteration >= max_iterations:
            return "workspace_prepared"

        # The first iteration must always prepare the workspace, regardless of
        # the (expectedly failing) evaluation results.
        if iteration == 0:
            return "structure_incomplete"

        if _evaluation_has_problems(state):
            return "structure_incomplete"

        # No errors and no failing results -> preparation is complete.
        return "workspace_prepared"

    return evaluate_preparation
