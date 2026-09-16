from unittest import TestCase

from mdeagent.mapping import mde_to_files
from mdeagent.state import MDEAgentState


class TestMDEToFiles(TestCase):
    def test_mapping(self):
        state = MDEAgentState(written_files=["file1.txt", "file2.txt"])
        evaluation_params = mde_to_files(state)
        self.assertIn("files", evaluation_params)
        self.assertEqual(evaluation_params["files"], ["file1.txt", "file2.txt"])
