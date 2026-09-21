"""Tests for the implementation agent loop conditional edge function.

These tests cover the requirements of the agent-loop refactoring:

1. The evaluation node was renamed to ``evaluate_implementation`` (see
   ``tests/implementation/build_implementation_graph_test.py``).
2. ``ImplementationState`` carries an ``iteration`` field (renamed from
   ``implementation_iteration``).
3. A conditional edge function routes ``integration_error`` ->
   ``implement_bx_tool`` and ``implementation_error`` ->
   ``implement_transformation``. The ``integration_error`` can only be
   returned when the bx tool integration is part of the graph
   (``integration_enabled=True``).
"""

from datetime import UTC, datetime
from unittest import TestCase

from mdeagent.evaluation.types import EvaluationError, EvaluationResult, EvaluationRun
from mdeagent.implementation.evaluation.evaluate_transformation_implementation import (
    create_evaluate_transformation_implementation,
)
from mdeagent.implementation.state import ImplementationState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _run(results=None, errors=None) -> EvaluationRun:
    return EvaluationRun(
        started_at=datetime.now(tz=UTC),
        execution_time_ms=0,
        iteration=1,
        results=results or [],
        errors=errors or [],
    )


def _result(success: bool) -> EvaluationResult:
    return EvaluationResult(
        content="some content",
        metadata={"success": success, "include_in_report": False},
    )


def _make_dummy_transformation_plan() -> object:
    """Create a minimal mock transformation plan for testing."""
    from unittest.mock import MagicMock
    from mdeagent.comprehension.plan import TransformationPlanData
    
    mock_tp = MagicMock()
    mock_tp.data = {
        "source_model_implementation": "",
        "target_model_implementation": "",
        "transformation_direction": "",
        "implementation_steps": "",
        "difficulties": "",
        "source_model_package": "",
        "target_model_package": "",
        "source_model_name": "",
        "target_model_name": "",
    }
    return mock_tp


def _state(iteration=1, latest_evaluation_runs=None, **kwargs) -> ImplementationState:
    """Build an ImplementationState with sensible defaults for the decision tests.
    
    Required fields that are not relevant for evaluation decisions are set to dummy values.
    Note: bxtool_path must be a Path per original state definition (not None).
    """
    from pathlib import Path
    from mdeagent.implementation.types import TransformationClass
    
    # Create minimal dummy objects for required fields
    dummy_path = Path("/tmp/dummy")
    dummy_tp = _make_dummy_transformation_plan()
    dummy_tc: TransformationClass = {
        "name": "DummyTransformation",
        "package": "com.example",
        "path": dummy_path,
        "code": None,
    }
    
    state = ImplementationState(
        transformation_plan=dummy_tp,  # type: ignore
        transformation_class=dummy_tc,
        task_specification="dummy task",
        maven_project_path=dummy_path,
        bxtool_path=dummy_path,  # Must be Path, not None
        written_files=[],
        latest_evaluation_runs=latest_evaluation_runs if latest_evaluation_runs is not None else {},
        iteration=iteration,
        **{k: v for k, v in kwargs.items() if k not in [
            'transformation_plan', 'transformation_class',
            'task_specification', 'maven_project_path', 'bxtool_path',
            'written_files', 'latest_evaluation_runs', 'iteration'
        ]},
    )
    return state


# --------------------------------------------------------------------------- #
# 3. Conditional edge decision function
# --------------------------------------------------------------------------- #
class TestEvaluateTransformationImplementation(TestCase):
    def test_decision__is_callable(self):
        decide = create_evaluate_transformation_implementation()
        self.assertTrue(callable(decide))

    # --- max_iteration_reached -------------------------------------------- #
    def test_decision__max_iteration_reached(self):
        """The safety guard must terminate the loop even with failing results."""
        MAX_ITERATIONS = 5
        decide = create_evaluate_transformation_implementation()
        state = _state(
            iteration=MAX_ITERATIONS,
            latest_evaluation_runs={
                "file_existence": _run(errors=[EvaluationError(message="boom", type="ValueError")]),
            },
        )
        self.assertEqual(
            decide(state, max_iterations=MAX_ITERATIONS),
            "max_iteration_reached",
        )

    def test_decision__custom_max_iterations(self):
        decide = create_evaluate_transformation_implementation()
        state = _state(
            iteration=2,
            latest_evaluation_runs={
                "file_existence": _run(errors=[EvaluationError(message="boom", type="ValueError")]),
            },
        )
        # iteration (2) >= max_iterations (2) -> terminate.
        self.assertEqual(decide(state, max_iterations=2), "max_iteration_reached")
        # iteration (2) < max_iterations (10) and there are problems -> retry.
        self.assertEqual(decide(state, max_iterations=10), "implementation_error")

    # --- implementation_error --------------------------------------------- #
    def test_decision__implementation_error_via_errors(self):
        decide = create_evaluate_transformation_implementation()
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "file_existence": _run(errors=[EvaluationError(message="boom", type="ValueError")]),
                "java_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_error")

    def test_decision__implementation_error_via_failing_result(self):
        """Implementation evaluations signal problems via results with success=False."""
        decide = create_evaluate_transformation_implementation()
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "file_existence": _run(results=[_result(success=False)]),
                "java_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_error")

    # --- integration_error ------------------------------------------------ #
    def test_decision__integration_error_when_integration_enabled(self):
        """Requirement 3: integration_error routes to implement_bx_tool."""
        decide = create_evaluate_transformation_implementation(integration_enabled=True)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "integration_compilation": _run(
                    errors=[EvaluationError(message="integration boom", type="ValueError")]
                ),
                "file_existence": _run(),
                "java_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "integration_error")

    def test_decision__integration_error_via_failing_result_when_enabled(self):
        decide = create_evaluate_transformation_implementation(integration_enabled=True)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "integration_compilation": _run(results=[_result(success=False)]),
                "file_existence": _run(),
                "java_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "integration_error")

    def test_decision__no_integration_error_when_integration_disabled(self):
        """Requirement 3: integration_error can only occur when the bx tool is integrated.

        When ``integration_enabled`` is ``False`` an integration_compilation
        failure must be reported as an ``implementation_error`` (routing back to
        ``implement_transformation``) instead.
        """
        decide = create_evaluate_transformation_implementation(integration_enabled=False)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "integration_compilation": _run(
                    errors=[EvaluationError(message="integration boom", type="ValueError")]
                ),
                "file_existence": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_error")

    def test_decision__missing_integration_run_is_not_an_integration_error(self):
        """When the integration run was not executed (None) it is not treated as
        an integration error on its own (robustness against a missing run)."""
        decide = create_evaluate_transformation_implementation(integration_enabled=True)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                # integration_compilation is absent
                "file_existence": _run(),
                "java_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_success")

    # --- implementation_success ------------------------------------------- #
    def test_decision__implementation_success(self):
        decide = create_evaluate_transformation_implementation(integration_enabled=True)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "file_existence": _run(),
                "java_compilation": _run(),
                "integration_compilation": _run(),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_success")

    def test_decision__success_with_clean_results_metadata(self):
        decide = create_evaluate_transformation_implementation()
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "file_existence": _run(results=[_result(success=True)]),
                "java_compilation": _run(results=[_result(success=True)]),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "implementation_success")

    def test_decision__integration_error_takes_precedence_when_enabled(self):
        """When integration is enabled an integration error is reported before
        an implementation error (so the bx tool is fixed first)."""
        decide = create_evaluate_transformation_implementation(integration_enabled=True)
        state = _state(
            iteration=1,
            latest_evaluation_runs={
                "integration_compilation": _run(
                    errors=[EvaluationError(message="integration boom", type="ValueError")]
                ),
                "file_existence": _run(errors=[EvaluationError(message="impl boom", type="ValueError")]),
            },
        )
        self.assertEqual(decide(state, max_iterations=5), "integration_error")


# --------------------------------------------------------------------------- #
# 2. iteration field is part of the state
# --------------------------------------------------------------------------- #
class TestImplementationStateIteration(TestCase):
    def test_state__iteration_field_exists(self):
        """Verify that the iteration field exists and can be set."""
        state = _state(iteration=3)
        self.assertEqual(state["iteration"], 3)

    def test_state__iteration_can_be_zero(self):
        """When creating a state for tests, iteration typically starts at 0."""
        state = _state(iteration=0)
        self.assertEqual(state["iteration"], 0)
