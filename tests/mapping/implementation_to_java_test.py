from unittest import TestCase

from mdeagent.implementation.state import ImplementationState
from mdeagent.mapping import implementation_to_java_files


class TestCodingToFileMapping(TestCase):
    def test_mapping(self):
        state = ImplementationState(written_files=["file1.java", "file2.java"])
        evaluation_params = implementation_to_java_files(state)
        self.assertIn("files", evaluation_params)
        self.assertEqual(evaluation_params["files"], ["file1.java", "file2.java"])
