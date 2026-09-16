"""Tests for the implementation agent graph (``build_implementation_graph``).

These tests cover the requirements of the agent-loop refactoring:

1. The evaluation node was renamed to ``evaluate_implementation``.
3. A conditional edge after ``evaluate_implementation`` routes
   ``integration_error`` -> ``implement_bx_tool`` (only when the bx tool is
   integrated) and ``implementation_error`` -> ``implement_transformation``.

The graph topology itself is *not* changed (unlike the preparation subgraph):
``implement_transformation`` -> (optional) ``implement_bx_tool`` ->
``format_code`` -> ``evaluate_implementation`` -> conditional edge.
"""

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import GraphOutput
from pydantic import BaseModel

from mdeagent.evaluation import EvaluationExecutor
from mdeagent.evaluation.types import (
    Evaluation,
    EvaluationError,
    EvaluationResult,
    EvaluationRun,
)
from mdeagent.implementation.agent import build_implementation_graph
from mdeagent.implementation.state import ImplementationState


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class _FilesSchema(BaseModel):
    """Permissive schema accepting the ``implementation_to_java_files`` mapper output."""

    files: list[Path] = []


class _StatefulEvaluation(Evaluation):
    """An evaluation whose ``run`` fails (returns a failing result) for the
    first ``fail_until`` calls and succeeds (no results / no errors) afterwards.

    This mirrors ``_StatefulEvaluation`` from the preparation tests so that the
    implementation loop can be exercised without the real (Maven/LLM) heavy
    evaluations.
    """

    def __init__(self, fail_until: int = 0):
        self._fail_until = fail_until
        self._calls = 0

    async def setup(self) -> None:
        pass

    async def run(self, **kwargs) -> tuple[list[EvaluationResult], list[EvaluationError]]:
        self._calls += 1
        if self._calls <= self._fail_until:
            return (
                [
                    EvaluationResult(
                        content="some content",
                        metadata={"success": False, "include_in_report": False},
                    )
                ],
                [],
            )
        return ([], [])


def _run(results=None, errors=None) -> EvaluationRun:
    return EvaluationRun(
        started_at=datetime.now(tz=UTC),
        execution_time_ms=0,
        iteration=1,
        results=results or [],
        errors=errors or [],
    )


def _build_executor(fail_until: int) -> EvaluationExecutor:
    """Executor with the same evaluation ids the implementation graph uses.

    ``file_existence`` and ``java_compilation`` share the same stateful
    evaluation instance so that ``register_linked_evaluation`` (called inside
    ``build_implementation_graph``) can wire ``integration_compilation`` to
    ``java_compilation``. Each graph cycle executes ``file_existence`` and
    ``java_compilation`` (two calls per cycle), so ``fail_until`` is expressed
    in number of calls.
    """
    shared = _StatefulEvaluation(fail_until=fail_until)
    return EvaluationExecutor(
        evaluations={
            "file_existence": {
                "evaluation": shared,
                "evaluation_schema": _FilesSchema,
            },
            "java_compilation": {
                "evaluation": shared,
                "evaluation_schema": _FilesSchema,
            },
        }
    )


# --------------------------------------------------------------------------- #
# 1. & 3. Graph construction and structure
# --------------------------------------------------------------------------- #
class TestBuildImplementationGraph(TestCase):
    @patch("mdeagent.implementation.agent.create_implement_transformation_node")
    @patch("mdeagent.implementation.agent.create_implement_bx_tool_node")
    @patch("mdeagent.implementation.agent.create_evaluation_node")
    @patch("mdeagent.implementation.agent.create_evaluate_transformation_implementation")
    @patch("mdeagent.implementation.agent.build_base_model")
    def test_build_implementation_graph__without_benchmarx(
        self,
        mock_build_base_model: Mock,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_node: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        # Mocks
        mock_build_base_model.return_value = Mock(name="base_model")
        mock_create_implement_transformation_node.return_value = Mock(name="impl_transformation")
        mock_create_evaluation_node.return_value = Mock(name="evaluate_implementation")
        mock_evaluate_transformation_implementation.return_value = Mock(name="decision")

        # Function under test (no benchmarx -> no implement_bx_tool node)
        graph = build_implementation_graph(
            evaluation_executor=Mock(name="evaluation_executor"),
            workspace_path=Mock(name="workspace_path"),
        )

        # Assertions
        self.assertIsInstance(graph, StateGraph)
        mock_evaluate_transformation_implementation.assert_called_once_with(
            integration_enabled=False
        )
        mock_create_evaluation_node.assert_called_once()
        # implement_bx_tool must NOT be created when benchmarx_path is None.
        mock_create_implement_bx_tool_node.assert_not_called()
        mock_create_implement_transformation_node.assert_called_once()

    @patch("mdeagent.implementation.agent.create_implement_transformation_node")
    @patch("mdeagent.implementation.agent.create_implement_bx_tool_node")
    @patch("mdeagent.implementation.agent.create_evaluation_node")
    @patch("mdeagent.implementation.agent.create_evaluate_transformation_implementation")
    @patch("mdeagent.implementation.agent.build_base_model")
    def test_build_implementation_graph__with_benchmarx(
        self,
        mock_build_base_model: Mock,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_node: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        mock_build_base_model.return_value = Mock(name="base_model")
        mock_create_implement_transformation_node.return_value = Mock(name="impl_transformation")
        mock_create_implement_bx_tool_node.return_value = Mock(name="impl_bx_tool")
        mock_create_evaluation_node.return_value = Mock(name="evaluate_implementation")
        mock_evaluate_transformation_implementation.return_value = Mock(name="decision")

        graph = build_implementation_graph(
            evaluation_executor=Mock(name="evaluation_executor"),
            workspace_path=Mock(name="workspace_path"),
            benchmarx_path=Path("/tmp/benchmarx"),
        )

        self.assertIsInstance(graph, StateGraph)
        mock_evaluate_transformation_implementation.assert_called_once_with(
            integration_enabled=True
        )
        mock_create_implement_bx_tool_node.assert_called_once()
        mock_create_implement_transformation_node.assert_called_once()

    @patch("mdeagent.implementation.agent.create_implement_transformation_node")
    @patch("mdeagent.implementation.agent.create_implement_bx_tool_node")
    @patch("mdeagent.implementation.agent.create_evaluation_node")
    @patch("mdeagent.implementation.agent.create_evaluate_transformation_implementation")
    @patch("mdeagent.implementation.agent.build_base_model")
    def test_build_implementation_graph__registers_linked_evaluation(
        self,
        mock_build_base_model: Mock,
        mock_evaluate_transformation_implementation: Mock,
        mock_create_evaluation_agent_work_function: Mock,
        mock_create_implement_bx_tool_node: Mock,
        mock_create_implement_transformation_node: Mock,
    ):
        mock_build_base_model.return_value = Mock(name="base_model")
        mock_create_implement_transformation_node.return_value = Mock(name="impl_transformation")
        mock_create_evaluation_agent_work_function.return_value = Mock(name="evaluate_implementation")
        mock_evaluate_transformation_implementation.return_value = Mock(name="decision")
        mock_evaluation_executor = Mock(name="evaluation_executor")

        build_implementation_graph(
            evaluation_executor=mock_evaluation_executor,
            workspace_path=Mock(name="workspace_path"),
        )

        mock_evaluation_executor.register_linked_evaluation.assert_called_once_with(
            "integration_compilation", "java_compilation"
        )


class TestImplementationGraphStructure(TestCase):
    """Requirement 1 (renamed node) & 3 (conditional edges)."""

    def _compile(self, benchmarx_path: Path | None = None) -> CompiledStateGraph:
        with patch("mdeagent.implementation.agent.build_base_model"), patch(
            "mdeagent.implementation.agent.create_implement_transformation_node",
            return_value=Mock(name="impl_transformation"),
        ), patch(
            "mdeagent.implementation.agent.create_format_code_node",
            return_value=Mock(name="format_code"),
        ):
            if benchmarx_path:
                with patch(
                    "mdeagent.implementation.agent.create_implement_bx_tool_node",
                    return_value=Mock(name="impl_bx_tool"),
                ):
                    return build_implementation_graph(
                        evaluation_executor=Mock(name="evaluation_executor"),
                        workspace_path=Mock(name="workspace_path"),
                        benchmarx_path=benchmarx_path,
                    ).compile()
            return build_implementation_graph(
                evaluation_executor=Mock(name="evaluation_executor"),
                workspace_path=Mock(name="workspace_path"),
            ).compile()

    def _edges(self, graph: CompiledStateGraph):
        return [
            (e.source, e.target, e.conditional, e.data)
            for e in graph.get_graph().edges
        ]

    def test_graph__node_renamed_to_evaluate_implementation(self):
        """Requirement 1: the evaluation node is called evaluate_implementation."""
        graph = self._compile()
        nodes = graph.get_graph().nodes
        self.assertIn("evaluate_implementation", nodes)
        self.assertIn("implement_transformation", nodes)
        self.assertIn("format_code", nodes)
        self.assertNotIn("evaluation_agentic_work", nodes)

    def test_graph__start_edges_to_implement_transformation(self):
        """The topology is unchanged: START -> implement_transformation first."""
        graph = self._compile()
        edges = self._edges(graph)
        self.assertIn(("__start__", "implement_transformation", False, None), edges)

    def test_graph__format_code_edges_to_evaluate_implementation(self):
        graph = self._compile()
        edges = self._edges(graph)
        self.assertIn(("format_code", "evaluate_implementation", False, None), edges)

    def test_graph__conditional_edge_without_benchmarx(self):
        """Requirement 3: without bx tool only implementation_error / END exist."""
        graph = self._compile()
        edges = self._edges(graph)
        conditional_targets = {
            t for s, t, cond, _ in edges if s == "evaluate_implementation" and cond
        }
        self.assertEqual(
            conditional_targets,
            {"implement_transformation", "__end__"},
            "evaluate_implementation must conditionally route to "
            "implement_transformation or END when no bx tool is integrated.",
        )
        # integration_error must NOT be routable when the bx tool is absent.
        end_labels = [
            data for s, t, cond, data in edges
            if s == "evaluate_implementation" and cond
        ]
        self.assertNotIn("integration_error", end_labels)
        self.assertIn("implementation_error", end_labels)

    def test_graph__conditional_edge_with_benchmarx(self):
        """Requirement 3: with bx tool integration_error -> implement_bx_tool."""
        graph = self._compile(benchmarx_path=Path("/tmp/benchmarx"))
        nodes = graph.get_graph().nodes
        self.assertIn("implement_bx_tool", nodes)
        edges = self._edges(graph)
        conditional_targets = {
            t for s, t, cond, _ in edges if s == "evaluate_implementation" and cond
        }
        self.assertEqual(
            conditional_targets,
            {"implement_transformation", "implement_bx_tool", "__end__"},
            "evaluate_implementation must conditionally route to "
            "implement_transformation, implement_bx_tool or END.",
        )
        labels = [
            data for s, t, cond, data in edges
            if s == "evaluate_implementation" and cond
        ]
        self.assertIn("integration_error", labels)
        self.assertIn("implementation_error", labels)

    def test_graph__bx_tool_inserted_between_transformation_and_format(self):
        graph = self._compile(benchmarx_path=Path("/tmp/benchmarx"))
        edges = self._edges(graph)
        self.assertIn(("implement_transformation", "implement_bx_tool", False, None), edges)
        self.assertIn(("implement_bx_tool", "format_code", False, None), edges)


# --------------------------------------------------------------------------- #
# End-to-end loop: evaluation fails -> implement_transformation retries -> success
# --------------------------------------------------------------------------- #
class TestImplementationLoop(TestCase):
    """Verifies that the conditional edge drives a real retry loop.

    The heavy ``implement_transformation`` (LLM) and ``format_code`` (Maven)
    nodes are replaced by lightweight mocks so the loop logic can be tested
    without mvn / LLM. The evaluation runs for real with a stateful mock that
    reports problems for the first cycle(s) and success afterwards. The
    conditional edge function is the real one (no mock).
    """

    def _build_mocks(self):
        calls = {"implement_transformation": 0, "format_code": 0}

        def mock_implement_transformation_node(*args, **kwargs):
            async def implement_transformation(state: ImplementationState):
                calls["implement_transformation"] += 1
                # The iteration counter is advanced by the (real) evaluation
                # node, not by the work nodes.
                return {}
            return implement_transformation

        def mock_format_code_node(*args, **kwargs):
            async def format_code(state: ImplementationState):
                calls["format_code"] += 1
                return {}
            return format_code

        return calls, mock_implement_transformation_node, mock_format_code_node

    def test_loop__retries_until_evaluation_succeeds(self):
        calls, mock_impl, mock_format = self._build_mocks()

        # Each graph cycle runs file_existence + java_compilation (they share
        # one stateful instance -> 2 calls per cycle). fail_until=2 means the
        # first cycle fails (calls 1, 2) and the second cycle succeeds
        # (calls 3, 4) -> exactly one retry.
        executor = _build_executor(fail_until=2)

        with patch(
            "mdeagent.implementation.agent.create_implement_transformation_node",
            side_effect=mock_impl,
        ), patch(
            "mdeagent.implementation.agent.create_format_code_node",
            side_effect=mock_format,
        ), patch(
            "mdeagent.implementation.agent.build_base_model"
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                graph = build_implementation_graph(
                    evaluation_executor=executor,
                    workspace_path=Path(temp_dir),
                ).compile()

                initial_state = ImplementationState(
                    maven_project_path=Path(temp_dir),
                    transformation_class_path=Path(temp_dir) / "Trans.java",
                    bxtool_path=Path(temp_dir) / "BxTool.java",
                    written_java_files=[],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: ImplementationState = output.value

        # implement_transformation is called twice: once for the initial cycle
        # and once for the retry triggered by the failing evaluation.
        self.assertEqual(
            calls["implement_transformation"],
            2,
            "implement_transformation should be called twice (initial + one retry).",
        )
        # The loop must terminate and end with iteration == 2.
        self.assertEqual(output_state.get("iteration"), 2)
        # The final evaluation reported no problems.
        latest = output_state.get("latest_evaluation_runs", {})
        for run in latest.values():
            self.assertEqual(run.errors, [])
            self.assertEqual(run.results, [])

    def test_loop__terminates_immediately_when_first_evaluation_is_clean(self):
        """When the first evaluation is clean the loop terminates after one
        implementation cycle (no retry)."""
        calls, mock_impl, mock_format = self._build_mocks()

        executor = _build_executor(fail_until=0)

        with patch(
            "mdeagent.implementation.agent.create_implement_transformation_node",
            side_effect=mock_impl,
        ), patch(
            "mdeagent.implementation.agent.create_format_code_node",
            side_effect=mock_format,
        ), patch(
            "mdeagent.implementation.agent.build_base_model"
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                graph = build_implementation_graph(
                    evaluation_executor=executor,
                    workspace_path=Path(temp_dir),
                ).compile()

                initial_state = ImplementationState(
                    maven_project_path=Path(temp_dir),
                    transformation_class_path=Path(temp_dir) / "Trans.java",
                    bxtool_path=Path(temp_dir) / "BxTool.java",
                    written_java_files=[],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: ImplementationState = output.value
        self.assertEqual(calls["implement_transformation"], 1)
        self.assertEqual(output_state.get("iteration"), 1)

    def test_loop__terminates_via_max_iterations_on_persistent_error(self):
        """The safety guard must terminate the loop when the evaluation keeps
        failing instead of looping forever."""
        calls, mock_impl, mock_format = self._build_mocks()

        # Keep failing forever; the max_iterations guard must stop the loop.
        executor = _build_executor(fail_until=1000)

        with patch(
            "mdeagent.implementation.agent.create_implement_transformation_node",
            side_effect=mock_impl,
        ), patch(
            "mdeagent.implementation.agent.create_format_code_node",
            side_effect=mock_format,
        ), patch(
            "mdeagent.implementation.agent.build_base_model"
        ), patch(
            "mdeagent.implementation.evaluate_transformation_implementation.WORKFLOW_MAX_ITERATIONS",
            3,
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                graph = build_implementation_graph(
                    evaluation_executor=executor,
                    workspace_path=Path(temp_dir),
                ).compile()

                initial_state = ImplementationState(
                    maven_project_path=Path(temp_dir),
                    transformation_class_path=Path(temp_dir) / "Trans.java",
                    bxtool_path=Path(temp_dir) / "BxTool.java",
                    written_java_files=[],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: ImplementationState = output.value
        # With max_iterations=3 the loop runs 3 implementation cycles (iteration
        # 1, 2, 3) and then terminates via max_iteration_reached on the 4th
        # evaluation (iteration == 3 >= 3).
        self.assertEqual(calls["implement_transformation"], 3)
        self.assertEqual(output_state.get("iteration"), 3)

    def test_loop__with_benchmarx_increments_iteration_once_per_cycle(self):
        """Regression test for the double-increment bug.

        With the bx tool integrated the graph runs *two* work nodes per cycle
        (``implement_transformation`` + ``implement_bx_tool``). Previously both
        work nodes incremented ``iteration``, so a single cycle bumped the
        counter twice (and an ``integration_error`` retry that bypassed
        ``implement_transformation`` did not bump it at all). The counter is now
        advanced once per cycle in the ``evaluate_implementation`` node, so two
        cycles must yield ``iteration == 2`` (not ``4``).
        """
        calls, mock_impl, mock_format = self._build_mocks()

        bx_calls = {"implement_bx_tool": 0}

        def mock_implement_bx_tool_node(*args, **kwargs):
            async def implement_bx_tool(state: ImplementationState):
                bx_calls["implement_bx_tool"] += 1
                return {}
            return implement_bx_tool

        # fail_until=2: the first cycle fails (calls 1, 2), the second succeeds
        # (calls 3, 4) -> exactly one retry.
        executor = _build_executor(fail_until=2)

        with patch(
            "mdeagent.implementation.agent.create_implement_transformation_node",
            side_effect=mock_impl,
        ), patch(
            "mdeagent.implementation.agent.create_implement_bx_tool_node",
            side_effect=mock_implement_bx_tool_node,
        ), patch(
            "mdeagent.implementation.agent.create_format_code_node",
            side_effect=mock_format,
        ), patch(
            "mdeagent.implementation.agent.build_base_model"
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                graph = build_implementation_graph(
                    evaluation_executor=executor,
                    workspace_path=Path(temp_dir),
                    benchmarx_path=Path(temp_dir) / "benchmarx",
                ).compile()

                initial_state = ImplementationState(
                    maven_project_path=Path(temp_dir),
                    transformation_class_path=Path(temp_dir) / "Trans.java",
                    bxtool_path=Path(temp_dir) / "BxTool.java",
                    written_java_files=[],
                    iteration=0,
                )
                output: GraphOutput = asyncio.run(
                    graph.ainvoke(input=initial_state, version="v2")
                )

        output_state: ImplementationState = output.value
        # Both work nodes ran in each of the two cycles.
        self.assertEqual(calls["implement_transformation"], 2)
        self.assertEqual(bx_calls["implement_bx_tool"], 2)
        # iteration must be 2 (one per cycle). If both work nodes still
        # incremented it, this would be 4.
        self.assertEqual(
            output_state.get("iteration"),
            2,
            "iteration must be advanced once per cycle (in the evaluation "
            "node), not once per work node.",
        )
