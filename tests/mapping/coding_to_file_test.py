from unittest import TestCase

from bxagent.tools.coding.state import CodingAgentState
from bxagent.mapping import map_coding_to_file

class TestCodingToFileMapping(TestCase):
    def test_mapping(self):
        state = CodingAgentState(written_java_files=["file1.java", "file2.java"])
        validation_params = map_coding_to_file(state)
        self.assertIn("files", validation_params)
        self.assertEqual(validation_params["files"], ["file1.java", "file2.java"])
