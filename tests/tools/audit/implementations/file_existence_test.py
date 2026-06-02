import asyncio

from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

from bxagent.tools.audit.implementations.file_existence import FileExistenceAudit


class TestFileExistence(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(FileExistenceAudit, "setup"),
            "FileExistenceAudit should have a 'setup' method.",
        )

        file_existence_audit = FileExistenceAudit(files=[])

        self.assertIsNone(
            asyncio.run(file_existence_audit.setup()),
            "FileExistenceAudit's 'setup' method should return None.",
        )

    def test_execute__method_defined(self):
        self.assertTrue(
            hasattr(FileExistenceAudit, "run"),
            "FileExistenceAudit should have a 'run' method.",
        )

    @patch("pathlib.Path.exists")
    def test_execute__files_exists(self, mock_exists):
        mock_exists.return_value = True
        files = [Path("file1.txt"), Path("file2.txt")]

        file_existence_audit = FileExistenceAudit(files=files)
        results, errors = asyncio.run(file_existence_audit.run())

        self.assertEqual(len(errors), 0, "There should be no errors when files exist.")

        for expected_file, result in zip(files, results):
            self.assertEqual(
                result.content,
                f"File exists: {expected_file}",
                f"Expected content for existing file {expected_file} does not match.",
            )

    @patch("pathlib.Path.exists")
    def test_execute__files_do_not_exist(self, mock_exists):
        mock_exists.return_value = False
        files = [Path("file1.txt"), Path("file2.txt")]

        file_existence_audit = FileExistenceAudit(files=files)
        results, errors = asyncio.run(file_existence_audit.run())

        self.assertEqual(
            len(results), 0, "There should be no results when files do not exist."
        )

        for expected_file, error in zip(files, errors):
            self.assertEqual(
                error.message,
                f"File does not exist: {expected_file}",
                f"Expected message for non-existing file {expected_file} does not match.",
            )
            self.assertDictEqual(
                error.details,
                {"file": str(expected_file)},
                f"Expected details for non-existing file {expected_file} do not match.",
            )

    @patch("pathlib.Path.exists", side_effect=[True, False])
    def test_execute__mixed_file_existence(self, mock_exists):
        files = [Path("file1.txt"), Path("file2.txt")]
        file_existence_audit = FileExistenceAudit(files=files)
        results, errors = asyncio.run(file_existence_audit.run())

        self.assertEqual(
            len(results), 1, "There should be one result when one file exists."
        )
        self.assertEqual(
            len(errors), 1, "There should be one error when one file does not exist."
        )
        self.assertEqual(
            mock_exists.call_count, 2, "exists should be called once per file."
        )

        self.assertEqual(
            results[0].content,
            f"File exists: {files[0]}",
            "Expected content for existing file does not match.",
        )
        self.assertEqual(
            errors[0].message,
            f"File does not exist: {files[1]}",
            "Expected message for non-existing file does not match.",
        )
        self.assertDictEqual(
            errors[0].details,
            {"file": str(files[1])},
            "Expected details for non-existing file do not match.",
        )
