from unittest import TestCase

from mdeagent.implementation.state import ImplementationState
from mdeagent.mapping import map_coding_to_file


class TestCodingToFileMapping(TestCase):
    def test_mapping(self):
        state = ImplementationState(written_java_files=["file1.java", "file2.java"])
        evaluation_params = map_coding_to_file(state)
        self.assertIn("files", evaluation_params)
        self.assertEqual(evaluation_params["files"], ["file1.java", "file2.java"])
