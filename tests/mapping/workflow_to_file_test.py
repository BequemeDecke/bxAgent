from unittest import TestCase

from bxagent.agents.workflow.state import WorkflowState
from bxagent.mapping import map_workflow_to_file

class TestWorkflowToFileMapping(TestCase):
    def test_mapping(self):
        state = WorkflowState(written_files=["file1.txt", "file2.txt"])
        audit_params = map_workflow_to_file(state)
        self.assertIn("files", audit_params)
        self.assertEqual(audit_params["files"], ["file1.txt", "file2.txt"])
