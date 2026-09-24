"""
Tests for the with_transformation utility function.

This module tests the state transformation wrapper that allows decoupling
state transformation logic from node execution logic.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest import TestCase

from mdeagent.util import with_transformation


# ==================== Basic Functionality Tests ====================


class TestWithTransformation__BasicFunctionality(TestCase):
    """Test basic transformation and execution behavior."""

    def test_with_transformation__applies_transform_before_node(self):
        """Transform should be applied first, then the node."""

        async def dummy_node(state):
            return {"node_field": state.get("transformed_value", "default")}

        def transform(state):
            return {"transformed_value": "hello", "extra_key": "extra_value"}

        wrapped = with_transformation(dummy_node, transform)
        result = asyncio.run(wrapped({}))

        self.assertEqual(result["transformed_value"], "hello")
        self.assertEqual(result["node_field"], "hello")
        self.assertEqual(result["extra_key"], "extra_value")

    def test_with_transformation__merges_transform_and_node_results(self):
        """Both transform and node results should be merged into the output."""

        async def dummy_node(state):
            return {"node_key": "node_value"}

        def transform(state):
            return {"transform_key": "transform_value"}

        wrapped = with_transformation(dummy_node, transform)
        result = asyncio.run(wrapped({}))

        self.assertEqual(result["transform_key"], "transform_value")
        self.assertEqual(result["node_key"], "node_value")

    def test_with_transformation__node_overrides_transform_on_conflict(self):
        """When transform and node set the same key, node value wins."""

        async def dummy_node(state):
            return {"shared_key": "from_node"}

        def transform(state):
            return {"shared_key": "from_transform", "only_transform": "value"}

        wrapped = with_transformation(dummy_node, transform)
        result = asyncio.run(wrapped({}))

        self.assertEqual(result["shared_key"], "from_node")
        self.assertEqual(result["only_transform"], "value")


# ==================== Path Transformation Tests ====================


class TestWithTransformation__PathTransformation(TestCase):
    """Test path normalization use cases (e.g., agent vs evaluator path differences)."""

    def test_with_transformation__strips_workspace_path_from_files(self):
        """Evaluator needs relative paths; transform converts absolute→relative.

        Scenario: previous nodes wrote absolute paths to state, this node
        (wrapped with transform) should operate on relative paths instead.
        """

        async def evaluator_node(state):
            # Evaluator reads relative paths (already converted by transform)
            return {"evaluated_count": len(state.get("written_files", []))}

        def strip_workspace(state):
            # Transform converts absolute → relative paths in state
            files = state.get("written_files", [])
            relative = [
                f.relative_to("/workspace") if str(f).startswith("/workspace") else f
                for f in files
            ]
            return {"written_files": relative}

        wrapped = with_transformation(evaluator_node, strip_workspace)

        # Initial state has absolute paths (written by previous agent nodes)
        initial_state = {
            "written_files": [
                Path("/workspace/src/Main.java"),
                Path("/workspace/src/Helper.java"),
            ]
        }
        result = asyncio.run(wrapped(initial_state))

        self.assertEqual(
            result["written_files"],
            [Path("src/Main.java"), Path("src/Helper.java")],
        )
        self.assertEqual(result["evaluated_count"], 2)

    def test_with_transformation__adds_workspace_path_back_for_evaluation(self):
        """Evaluator needs absolute paths; agent returns relative paths."""

        async def evaluation_node(state):
            # Evaluator node expects absolute paths
            paths = state.get("evaluation_paths", [])
            return {"evaluated": len(paths)}

        def add_workspace(state):
            # Transform adds workspace prefix to paths
            paths = state.get("evaluation_paths", [])
            return {
                "evaluation_paths": [
                    Path("/workspace") / p if not str(p).startswith("/") else p
                    for p in paths
                ]
            }

        wrapped = with_transformation(
            evaluation_node,
            add_workspace,
        )
        result = asyncio.run(
            wrapped({"evaluation_paths": [Path("src/Main.java")]})
        )

        self.assertEqual(
            result["evaluation_paths"],
            [Path("/workspace/src/Main.java")],
        )
        self.assertEqual(result["evaluated"], 1)


# ==================== Iteration Control Tests ====================


class TestWithTransformation__IterationControl(TestCase):
    """Test iteration incrementing use case."""

    def test_with_transformation__increments_iteration_counter(self):
        """Should increment iteration counter before node execution."""

        async def incrementing_node(state):
            return {"last_iteration_seen": state.get("iteration", 0)}

        def increment_iteration(state):
            return {"iteration": state.get("iteration", 0) + 1}

        wrapped = with_transformation(incrementing_node, increment_iteration)

        # First call: iteration 0 → 1
        result1 = asyncio.run(wrapped({"iteration": 0}))
        self.assertEqual(result1["iteration"], 1)
        self.assertEqual(result1["last_iteration_seen"], 1)

        # Second call: iteration 1 → 2
        result2 = asyncio.run(wrapped({"iteration": 1}))
        self.assertEqual(result2["iteration"], 2)
        self.assertEqual(result2["last_iteration_seen"], 2)

    def test_with_transformation__iteration_with_none_value(self):
        """Should handle missing iteration field gracefully."""

        async def node(state):
            return {"iteration": state.get("iteration", 0)}

        def increment(state):
            return {"iteration": state.get("iteration", 0) + 1}

        wrapped = with_transformation(node, increment)
        result = asyncio.run(wrapped({}))

        self.assertEqual(result["iteration"], 1)


# ==================== Chaining Tests ====================


class TestWithTransformation__Chaining(TestCase):
    """Test multiple transformations chained together."""

    def test_with_transformation__chains_multiple_transforms(self):
        """Multiple with_transformation wrappers should compose correctly."""

        async def leaf_node(state):
            return {"final": state.get("counter", 0)}

        def add_one(state):
            return {"counter": state.get("counter", 0) + 1}

        def multiply_two(state):
            return {"counter": state.get("counter", 0) * 2}

        # chain: multiply first, then add, then node
        first = with_transformation(leaf_node, add_one)
        second = with_transformation(first, multiply_two)

        result = asyncio.run(second({"counter": 10}))

        # multiply: 10 * 2 = 20
        # add: 20 + 1 = 21
        # node: returns {"final": 21}
        # merge: {"counter": 21, "final": 21}
        self.assertEqual(result["counter"], 21)
        self.assertEqual(result["final"], 21)


# ==================== Edge Case Tests ====================


class TestWithTransformation__EdgeCases(TestCase):
    """Test edge cases and error scenarios."""

    def test_with_transformation__empty_state(self):
        """Should handle empty state dict."""

        async def node(state):
            return {"result": "ok"}

        def identity_transform(state):
            return {}

        wrapped = with_transformation(node, identity_transform)
        result = asyncio.run(wrapped({}))

        self.assertEqual(result["result"], "ok")

    def test_with_transformation__returns_dict_not_state(self):
        """Should return only the merged dict, not the full state."""

        async def node(state):
            return {"node_only": True}

        def transform(state):
            return {"transform_only": True}

        wrapped = with_transformation(node, transform)
        result = asyncio.run(wrapped({"irrelevant": "value"}))

        # Should NOT contain the original irrelevant key
        self.assertNotIn("irrelevant", result)
        self.assertIn("node_only", result)
        self.assertIn("transform_only", result)


# ==================== Integration with MDEAgentState ====================


class TestWithTransformation__MDEAgentState(TestCase):
    """Test with actual MDEAgentState typing."""

    def test_with_transformation__preserves_type_safety(self):
        """Should work with typed state dicts."""
        from mdeagent.state import MDEAgentState

        async def path_transformer_node(state: MDEAgentState):
            workspace = state.get("workspace_path")
            return {"resolved_path": str(workspace) if workspace else "none"}

        def normalize_workspace(state: MDEAgentState):
            ws = state.get("workspace_path")
            if ws:
                return {"workspace_path": Path(str(ws)).resolve()}
            return {}

        wrapped = with_transformation(
            path_transformer_node, normalize_workspace
        )
        initial: MDEAgentState = {
            "workspace_path": Path("/tmp/../tmp/test"),
            "source_model_path": Path("/models/source"),
        }

        result = asyncio.run(wrapped(initial))

        # Workspace should be resolved — accept either /tmp or /private/tmp
        self.assertIn(str(result["workspace_path"]), ["/tmp/test", "/private/tmp/test"])
        self.assertIn(result["resolved_path"], ["/tmp/test", "/private/tmp/test"])


if __name__ == "__main__":
    import unittest
    unittest.main()
