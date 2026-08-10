from unittest import TestCase

from bxagent.agents.workflow.state import WorkflowState
from bxagent.mapping import map_workflow_to_file

class TestWorkflowToFileMapping(TestCase):
    def test_mapping(self):
        state = WorkflowState(written_files=["file1.txt", "file2.txt"])
        evaluation_params = map_workflow_to_file(state)
        self.assertIn("files", evaluation_params)
        self.assertEqual(evaluation_params["files"], ["file1.txt", "file2.txt"])
