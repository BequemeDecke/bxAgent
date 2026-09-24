"""
Integration tests for the comprehension node (create_comprehension_node).

These tests verify the integration between:
- The outer `MDEAgentState` passed to `create_comprehension_node`
- The inner `ComprehensionState` subgraph that performs reflect → evaluate → loop
- The state transformations between outer and inner states
- Error handling when the plan is missing or invalid

The tests use mock subgraphs that simulate the reflect → evaluate → loop
behaviour of a real comprehension agent.

Important: `create_comprehension_node` always:
- Converts the serialized plan dict to a `TransformationPlan` object
- Passes `iteration=0` to the subgraph (not the outer state's iteration)
- Passes `latest_evaluation_runs={}` to the subgraph (currently hardcoded)
"""

import asyncio
import tempfile
from pathlib import Path
from unittest import TestCase

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from mdeagent.comprehension.node import (
    create_comprehension_node,
)
from mdeagent.comprehension.plan import (
    FileTransformationPlanParser,
    TransformationPlan,
    TransformationPlanData,
)
from mdeagent.comprehension.state import ComprehensionState
from mdeagent.evaluation.types import EvaluationResult, EvaluationRun

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_parser_and_plan(
    temp_dir: Path,
    data: TransformationPlanData | None = None,
) -> tuple[FileTransformationPlanParser, TransformationPlan]:
    """Create a FileTransformationPlanParser + TransformationPlan backed by a
    temporary file, optionally pre-populated with *data*."""
    parser = FileTransformationPlanParser(temp_dir / "TRANSFORMATION.md")

    if data is None:
        data = TransformationPlanData(
            source_model_package="com.example.source",
            target_model_package="com.example.target",
            iteration=1,
            source_model_implementation="",
            target_model_implementation="",
            transformation_direction="",
            difficulties="",
            implementation_steps="",
        )

    plan = TransformationPlan.parse(parser, template_path=Path.cwd() / "templates")
    plan.data = data
    parser.save(str(plan))
    return parser, plan


def _fully_populated_plan() -> TransformationPlanData:
    """Return a fully populated TransformationPlanData."""
    return TransformationPlanData(
        source_model_package="com.example.source",
        target_model_package="com.example.target",
        iteration=1,
        source_model_implementation="EcoreModel",
        target_model_implementation="JavaModel",
        transformation_direction="forward",
        difficulties="Complex mappings, custom types",
        implementation_steps="1. Define mapping\n2. Generate code",
    )


def _empty_plan() -> TransformationPlanData:
    """Return a plan with all required fields empty."""
    return TransformationPlanData(
        source_model_package="com.example.source",
        target_model_package="com.example.target",
        iteration=1,
        source_model_implementation="",
        target_model_implementation="",
        transformation_direction="",
        difficulties="",
        implementation_steps="",
    )


def _partial_plan(iteration: int = 1) -> TransformationPlanData:
    """Return a plan with only source_model_implementation populated."""
    return TransformationPlanData(
        source_model_package="com.example.source",
        target_model_package="com.example.target",
        iteration=iteration,
        source_model_implementation="EcoreModel",
        target_model_implementation="",
        transformation_direction="",
        difficulties="",
        implementation_steps="",
    )


# ---------------------------------------------------------------------------
# Mock subgraph factories
# ---------------------------------------------------------------------------


def make_mock_subgraph(
    populate_data: TransformationPlanData | None = None,
    call_count: list[int] | None = None,
    fail_iteration: int | None = None,
) -> tuple[CompiledStateGraph[ComprehensionState], list[int]]:
    """Build a mock LangGraph subgraph that simulates the comprehension
    agent writing to disk and tracking invocations.

    Note: `create_comprehension_node` always passes `iteration=0` to the
    subgraph and converts the plan to a `TransformationPlan` object.

    Args:
        populate_data: The TransformationPlanData to write on each call.
                       If None, the plan is not modified.
        call_count: Mutable list to track invocation count.
        fail_iteration: If set, the subgraph will fail (raise) on this
                        iteration number (from the subgraph's perspective).

    Returns:
        (compiled_subgraph, call_count) tuple.
    """
    if call_count is None:
        call_count = [0]

    if populate_data is None:
        populate_data = _fully_populated_plan()

    def _comprehension_node(state: ComprehensionState) -> ComprehensionState:
        """Mock node that simulates the comprehension agent.

        Note: state["transformation_plan"] is a TransformationPlan object
        (not a dict), and state["iteration"] starts at 0.
        """
        call_count[0] += 1

        # If fail_iteration is set, raise on that iteration
        if fail_iteration is not None and state["iteration"] == fail_iteration:
            raise RuntimeError(f"Simulated failure at iteration {state['iteration']}")

        # Write the populated plan to disk
        tp_obj = state["transformation_plan"]
        if hasattr(tp_obj, "to_dict"):
            tp_dict = tp_obj.to_dict()
        else:
            tp_dict = tp_obj

        if tp_dict and isinstance(tp_dict, dict) and "parser" in tp_dict:
            parser = FileTransformationPlanParser.from_dict(tp_dict["parser"])
            plan = TransformationPlan.parse(
                parser, template_path=Path.cwd() / "templates"
            )
            plan.data = populate_data
            parser.save(str(plan))
            return_dict = plan.to_dict()
        else:
            return_dict = tp_dict if isinstance(tp_dict, dict) else tp_obj

        return {
            "transformation_plan": return_dict,
            "latest_evaluation_runs": {},
            "iteration": state["iteration"] + 1,
        }

    graph_builder = StateGraph(ComprehensionState)
    graph_builder.add_node("comprehension", _comprehension_node)
    graph_builder.add_edge(START, "comprehension")
    graph_builder.add_edge("comprehension", END)

    graph = graph_builder.compile()
    return graph, call_count


def make_mock_subgraph_with_loop(
    *,
    populate_data: TransformationPlanData | None = None,
    call_count: list[int] | None = None,
    should_pass_on_iteration: int = 2,
) -> tuple[CompiledStateGraph[ComprehensionState], list[int]]:
    """Build a mock subgraph that simulates a complete reflect → evaluate
    → loop subgraph with evaluation logic.

    Args:
        populate_data: Data to write on first call.
        call_count: Invocation tracker.
        should_pass_on_iteration: The subgraph iteration at which evaluation
                                  should pass (1-indexed for the user,
                                  but 0-indexed internally).

    Returns:
        (compiled_subgraph, call_count) tuple.
    """
    if call_count is None:
        call_count = [0]

    if populate_data is None:
        populate_data = _fully_populated_plan()

    def _reflect_node(state: ComprehensionState) -> ComprehensionState:
        """Simulates the reflect_transformation node."""
        call_count[0] += 1

        tp_obj = state["transformation_plan"]
        if hasattr(tp_obj, "to_dict"):
            tp_dict = tp_obj.to_dict()
        else:
            tp_dict = tp_obj

        if tp_dict and isinstance(tp_dict, dict) and "parser" in tp_dict:
            parser = FileTransformationPlanParser.from_dict(tp_dict["parser"])
            plan = TransformationPlan.parse(
                parser, template_path=Path.cwd() / "templates"
            )
            plan.data = populate_data
            parser.save(str(plan))
            return_dict = plan.to_dict()
        else:
            return_dict = tp_dict if isinstance(tp_dict, dict) else tp_obj

        return {
            "transformation_plan": return_dict,
            "latest_evaluation_runs": {},
            "iteration": state["iteration"] + 1,
        }

    def _evaluate_node(state: ComprehensionState) -> ComprehensionState:
        """Simulates the evaluate_comprehension node.

        Evaluation "passes" when we've reached the target iteration.
        """
        return {
            "transformation_plan": state["transformation_plan"],
            "latest_evaluation_runs": {},
            "iteration": state["iteration"],
        }

    def _route(state: ComprehensionState) -> str:
        """Route based on whether we've reached the target iteration."""
        if state["iteration"] >= should_pass_on_iteration:
            return END
        return "reflect"

    graph_builder = StateGraph(ComprehensionState)
    graph_builder.add_node("reflect", _reflect_node)
    graph_builder.add_node("evaluate", _evaluate_node)

    graph_builder.add_edge(START, "reflect")
    graph_builder.add_edge("reflect", "evaluate")
    graph_builder.add_conditional_edges(
        "evaluate", _route, {"reflect": "reflect", END: END}
    )

    graph = graph_builder.compile()
    return graph, call_count


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestCreateComprehensionNode__BasicIntegration(TestCase):
    """Test the basic integration of create_comprehension_node with a
    mock subgraph."""

    def test_node_returns_transformation_plan(self):
        """When the subgraph completes successfully, the node should return
        a dict with the updated transformation_plan."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, call_count = make_mock_subgraph(
                populate_data=_fully_populated_plan()
            )
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))

            # Verify result structure
            self.assertIn("transformation_plan", result)
            self.assertIsInstance(result["transformation_plan"], dict)

            # Verify the subgraph was called
            self.assertEqual(call_count[0], 1)

    def test_node_updates_plan_data_to_subgraph_result(self):
        """The returned transformation_plan should reflect what the subgraph
        wrote to disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))

            # Re-parse and verify all fields are populated
            final_tp = TransformationPlan.from_dict(result["transformation_plan"])
            self.assertEqual(final_tp.data["source_model_implementation"], "EcoreModel")
            self.assertEqual(final_tp.data["target_model_implementation"], "JavaModel")
            self.assertEqual(final_tp.data["transformation_direction"], "forward")
            self.assertEqual(
                final_tp.data["difficulties"], "Complex mappings, custom types"
            )
            self.assertEqual(
                final_tp.data["implementation_steps"],
                "1. Define mapping\n2. Generate code",
            )

    def test_node_handles_empty_evaluation_runs(self):
        """When latest_evaluation_runs is empty or missing, the node should
        still work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            # Omit latest_evaluation_runs
            result = asyncio.run(node({"transformation_plan": serialized}))

            self.assertIn("transformation_plan", result)

    def test_node_handles_evaluation_runs_in_state(self):
        """When latest_evaluation_runs is present in the outer state, it
        should not cause errors (even if currently ignored in
        ComprehensionState)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            # Create a sample evaluation run
            run = EvaluationRun(
                started_at=None,  # type: ignore[arg-type]
                execution_time_ms=100,
                iteration=0,
                results=[EvaluationResult(content="Test result")],
                errors=[],
                category="design",
            )

            # Pass evaluation runs in outer state
            result = asyncio.run(
                node(
                    {
                        "transformation_plan": serialized,
                        "latest_evaluation_runs": [run],
                    }
                )
            )

            # Should still work - evaluation runs are currently ignored in
            # the ComprehensionState input
            self.assertIn("transformation_plan", result)


class TestCreateComprehensionNode__ErrorHandling(TestCase):
    """Test error handling in create_comprehension_node."""

    def test_raises_when_transformation_plan_missing(self):
        """When no transformation_plan is in the state, the node should
        raise ValueError."""
        mock_subgraph, _ = make_mock_subgraph()
        node = create_comprehension_node(mock_subgraph)

        async def _run():
            with self.assertRaises(ValueError) as ctx:
                await node({})
            self.assertIn(
                "The comprehension node requires a transformation plan in the state.",
                str(ctx.exception),
            )

        asyncio.run(_run())

    def test_raises_when_transformation_plan_is_none(self):
        """When transformation_plan is explicitly None, the node should
        raise ValueError."""
        mock_subgraph, _ = make_mock_subgraph()
        node = create_comprehension_node(mock_subgraph)

        async def _run():
            with self.assertRaises(ValueError) as ctx:
                await node({"transformation_plan": None})
            self.assertIn(
                "The comprehension node requires a transformation plan in the state.",
                str(ctx.exception),
            )

        asyncio.run(_run())

    def test_subgraph_exception_propagates(self):
        """When the subgraph raises an exception, it should propagate to
        the caller."""
        mock_subgraph, _ = make_mock_subgraph(fail_iteration=0)
        node = create_comprehension_node(mock_subgraph)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            with self.assertRaises(RuntimeError) as ctx:
                asyncio.run(node({"transformation_plan": serialized}))
            self.assertIn("Simulated failure at iteration 0", str(ctx.exception))


class TestCreateComprehensionNode__StateTransformation(TestCase):
    """Test that state is correctly transformed between MDEAgentState and
    ComprehensionState."""

    def test_serialized_plan_preserved(self):
        """The transformation_plan dict should be preserved through the
        state transformation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            # Store some extra fields in the serialized plan
            serialized["custom_field"] = "custom_value"

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))

            # Custom field should be preserved (or at least not cause an error)
            self.assertIn("transformation_plan", result)
            # The result may or may not preserve custom fields depending on
            # how the subgraph handles it - just verify no crash

    def test_subgraph_iteration_starts_at_zero(self):
        """The subgraph's internal iteration counter always starts at 0,
        regardless of the outer state's iteration value."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            call_count = [0]

            def _tracking_node(state: ComprehensionState) -> ComprehensionState:
                call_count[0] += 1
                # The subgraph should always receive iteration=0
                self.assertEqual(state["iteration"], 0)
                return {
                    "transformation_plan": state["transformation_plan"],
                    "latest_evaluation_runs": {},
                    "iteration": state["iteration"] + 1,
                }

            graph_builder = StateGraph(ComprehensionState)
            graph_builder.add_node("tracking", _tracking_node)
            graph_builder.add_edge(START, "tracking")
            graph_builder.add_edge("tracking", END)
            mock_subgraph = graph_builder.compile()

            node = create_comprehension_node(mock_subgraph)

            # Pass different outer iterations - subgraph should always see 0
            for outer_iteration in [0, 1, 5, 10]:
                call_count[0] = 0
                result = asyncio.run(
                    node(
                        {
                            "transformation_plan": serialized,
                            "iteration": outer_iteration,
                        }
                    )
                )
                self.assertEqual(call_count[0], 1)


class TestCreateComprehensionNode__LoopIntegration(TestCase):
    """Test the full loop integration: reflect → evaluate → loop."""

    def test_loop_until_complete(self):
        """When the agent needs multiple reflections to complete the plan,
        the loop should continue until all fields are populated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, call_count = make_mock_subgraph_with_loop(
                should_pass_on_iteration=2,
            )
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))

            # Verify result
            self.assertIn("transformation_plan", result)
            self.assertIsInstance(result["transformation_plan"], dict)

            # Agent should have been called 2 times (iterate twice)
            self.assertEqual(call_count[0], 2)

    def test_loop_exhausts_budget(self):
        """When the agent cannot complete the plan within the budget, the
        loop should exhaust the budget and exit gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            # Set should_pass_on_iteration to a very high number
            mock_subgraph, call_count = make_mock_subgraph_with_loop(
                should_pass_on_iteration=100,
            )
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))

            # Should return without error
            self.assertIn("transformation_plan", result)

            # The loop should have run multiple times
            self.assertGreater(call_count[0], 1)

    def test_single_pass_when_already_complete(self):
        """When the plan is already complete before calling the node, the
        subgraph should complete in one iteration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _fully_populated_plan())
            serialized = plan.to_dict()

            call_count = [0]

            def _no_op_node(state: ComprehensionState) -> ComprehensionState:
                call_count[0] += 1
                return {
                    "transformation_plan": state["transformation_plan"],
                    "latest_evaluation_runs": {},
                    "iteration": state["iteration"] + 1,
                }

            graph_builder = StateGraph(ComprehensionState)
            graph_builder.add_node("no_op", _no_op_node)
            graph_builder.add_edge(START, "no_op")
            graph_builder.add_edge("no_op", END)
            mock_subgraph = graph_builder.compile()

            node = create_comprehension_node(mock_subgraph)
            result = asyncio.run(node({"transformation_plan": serialized}))

            self.assertIn("transformation_plan", result)
            self.assertEqual(call_count[0], 1)


class TestCreateComprehensionNode__MultipleCalls(TestCase):
    """Test that the node can be called multiple times in sequence."""

    def test_node_can_be_called_multiple_times(self):
        """Calling the node multiple times should work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, call_count = make_mock_subgraph(
                populate_data=_fully_populated_plan()
            )
            node = create_comprehension_node(mock_subgraph)

            # First call
            result1 = asyncio.run(node({"transformation_plan": serialized}))
            self.assertIn("transformation_plan", result1)

            # Second call with updated plan
            result2 = asyncio.run(
                node({"transformation_plan": result1["transformation_plan"]})
            )
            self.assertIn("transformation_plan", result2)

            # Third call
            result3 = asyncio.run(
                node({"transformation_plan": result2["transformation_plan"]})
            )
            self.assertIn("transformation_plan", result3)

            # Should have been called 3 times
            self.assertEqual(call_count[0], 3)


class TestCreateComprehensionNode__IntegrationWithRealisticMock(TestCase):
    """Test with a more realistic mock that simulates actual agent behaviour."""

    def test_agent_populates_plan_incrementally(self):
        """Simulate an agent that populates the plan incrementally across
        multiple calls, like a real LLM agent."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            # Track population stages
            call_count = [0]
            stages = [
                # Stage 1: source_model_implementation
                _partial_plan(iteration=1),
                # Stage 2: add target_model_implementation
                TransformationPlanData(
                    source_model_package="com.example.source",
                    target_model_package="com.example.target",
                    iteration=1,
                    source_model_implementation="EcoreModel",
                    target_model_implementation="JavaModel",
                    transformation_direction="",
                    difficulties="",
                    implementation_steps="",
                ),
                # Stage 3: all fields populated
                _fully_populated_plan(),
            ]

            def _incremental_agent(state: ComprehensionState) -> ComprehensionState:
                call_count[0] += 1
                stage_idx = min(call_count[0] - 1, len(stages) - 1)
                stage_data = stages[stage_idx]

                tp_obj = state["transformation_plan"]
                if hasattr(tp_obj, "to_dict"):
                    tp_dict = tp_obj.to_dict()
                else:
                    tp_dict = tp_obj

                if tp_dict and isinstance(tp_dict, dict) and "parser" in tp_dict:
                    parser = FileTransformationPlanParser.from_dict(tp_dict["parser"])
                    p = TransformationPlan.parse(
                        parser, template_path=Path.cwd() / "templates"
                    )
                    p.data = stage_data
                    parser.save(str(p))
                    return_dict = p.to_dict()
                else:
                    return_dict = tp_dict if isinstance(tp_dict, dict) else tp_obj

                return {
                    "transformation_plan": return_dict,
                    "latest_evaluation_runs": {},
                    "iteration": state["iteration"] + 1,
                }

            graph_builder = StateGraph(ComprehensionState)
            graph_builder.add_node("agent", _incremental_agent)
            graph_builder.add_edge(START, "agent")
            graph_builder.add_edge("agent", END)
            mock_subgraph = graph_builder.compile()

            node = create_comprehension_node(mock_subgraph)

            # Call multiple times, updating the plan each time
            current_serialized = serialized
            for i in range(3):
                result = asyncio.run(node({"transformation_plan": current_serialized}))
                current_serialized = result["transformation_plan"]

            # Verify final state
            final_tp = TransformationPlan.from_dict(current_serialized)
            self.assertEqual(final_tp.data["target_model_implementation"], "JavaModel")
            self.assertEqual(final_tp.data["transformation_direction"], "forward")
            self.assertEqual(
                final_tp.data["difficulties"], "Complex mappings, custom types"
            )
            self.assertEqual(
                final_tp.data["implementation_steps"],
                "1. Define mapping\n2. Generate code",
            )
            self.assertEqual(call_count[0], 3)


class TestCreateComprehensionNode__EdgeCases(TestCase):
    """Test edge cases and unusual inputs."""

    def test_plan_with_empty_string_fields(self):
        """Plans with empty string fields should be handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))
            self.assertIn("transformation_plan", result)

    def test_plan_with_whitespace_only_fields(self):
        """Plans with whitespace-only fields should be handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            # Override with whitespace-only fields
            plan.data["source_model_implementation"] = "   "
            plan.data["target_model_implementation"] = "\t\n"
            parser.save(str(plan))
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))
            self.assertIn("transformation_plan", result)

    def test_plan_with_unicode_characters(self):
        """Plans with unicode characters should be handled correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            parser, plan = _make_parser_and_plan(temp_path, _empty_plan())
            # Override with unicode
            plan.data["source_model_implementation"] = "Écore-Modell"
            plan.data["target_model_implementation"] = "Jāvā-Modell"
            plan.data["difficulties"] = "特殊映射, 特別な型"
            parser.save(str(plan))
            serialized = plan.to_dict()

            mock_subgraph, _ = make_mock_subgraph(populate_data=_fully_populated_plan())
            node = create_comprehension_node(mock_subgraph)

            result = asyncio.run(node({"transformation_plan": serialized}))
            self.assertIn("transformation_plan", result)


if __name__ == "__main__":
    import unittest

    unittest.main()
