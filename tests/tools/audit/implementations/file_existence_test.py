import unittest

from src.tools.audit.implementations.file_existence import FileExistenceAudit
from pathlib import Path


class TestFileExistence(unittest.TestCase):
    def setUp(self):
        self.files = [Path("/path/to/file1.txt"), Path("/path/to/file2.txt")]
        self.file_existence_audit = FileExistenceAudit(files=self.files)

    def test_setup(self):
        self.assertTrue(
            hasattr(FileExistenceAudit, "setup"),
            "FileExistenceAudit should have a 'setup' method.",
        )

    def test_execute(self):
        self.assertTrue(
            hasattr(FileExistenceAudit, "run"),
            "FileExistenceAudit should have a 'run' method.",
        )
