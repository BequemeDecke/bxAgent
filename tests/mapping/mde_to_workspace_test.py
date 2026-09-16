from pathlib import Path
from unittest import TestCase

from mdeagent.mapping import mde_to_workspace
from mdeagent.state import MDEAgentState


class TestMDEToWorkspace(TestCase):
    def test_mapping(self):
        state = MDEAgentState(
            workspace_path=Path("/path/to/workspace"),
            artifact_id="my-artifact",
            transformation_package_path="com.example.transformation",
        )
        evaluation_params = mde_to_workspace(state)
        self.assertIn("workspace_path", evaluation_params)
        self.assertEqual(evaluation_params["workspace_path"], state["workspace_path"])
        self.assertIn("artifact_id", evaluation_params)
        self.assertEqual(evaluation_params["artifact_id"], state["artifact_id"])
        self.assertIn("package_path", evaluation_params)
        self.assertEqual(
            evaluation_params["package_path"], state["transformation_package_path"]
        )
