from unittest import TestCase
from pathlib import Path

from bxagent.implementation.state import (
    CodingAgentState,
)  # Fix cyclic import from __init__ files
from bxagent.agents.workflow.state import WorkflowState
from bxagent.mapping import map_workflow_to_workspace


class TestWorkflowToWorkspaceMapping(TestCase):
    def test_mapping(self):
        state = WorkflowState(
            workspace_path=Path("/path/to/workspace"),
            transformation_package_path="com.example.transformation",
        )
        validation_params = map_workflow_to_workspace(state)
        self.assertIn("workspace_path", validation_params)
        self.assertEqual(validation_params["workspace_path"], state["workspace_path"])
        self.assertIn("package_path", validation_params)
        self.assertEqual(
            validation_params["package_path"], state["transformation_package_path"]
        )
