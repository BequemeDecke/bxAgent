from unittest import TestCase
from pathlib import Path

from mdeagent.implementation.state import (
    ImplementationState,
)  # Fix cyclic import from __init__ files
from mdeagent.agents.workflow.state import WorkflowState
from mdeagent.mapping import map_workflow_to_workspace


class TestWorkflowToWorkspaceMapping(TestCase):
    def test_mapping(self):
        state = WorkflowState(
            workspace_path=Path("/path/to/workspace"),
            transformation_package_path="com.example.transformation",
        )
        evaluation_params = map_workflow_to_workspace(state)
        self.assertIn("workspace_path", evaluation_params)
        self.assertEqual(evaluation_params["workspace_path"], state["workspace_path"])
        self.assertIn("package_path", evaluation_params)
        self.assertEqual(
            evaluation_params["package_path"], state["transformation_package_path"]
        )
