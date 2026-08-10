import asyncio

from unittest import TestCase
from unittest.mock import patch
from pathlib import Path

from bxagent.evaluation.implementations.file_existence import FileExistenceValidation


class TestFileExistence(TestCase):
    def test_setup__do_nothing(self):
        self.assertTrue(
            hasattr(FileExistenceValidation, "setup"),
            "FileExistenceValidation should have a 'setup' method.",
        )

        file_existence_validation = FileExistenceValidation()

        self.assertIsNone(
            asyncio.run(file_existence_validation.setup()),
            "FileExistenceValidation's 'setup' method should return None.",
        )

    def test_execute__method_defined(self):
        self.assertTrue(
            hasattr(FileExistenceValidation, "run"),
            "FileExistenceValidation should have a 'run' method.",
        )

    @patch("pathlib.Path.exists")
    def test_execute__files_exists(self, mock_exists):
        mock_exists.return_value = True
        files = [Path("file1.txt"), Path("file2.txt")]

        file_existence_validation = FileExistenceValidation()
        results, errors = asyncio.run(file_existence_validation.run(files=files))

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

        file_existence_validation = FileExistenceValidation()
        results, errors = asyncio.run(file_existence_validation.run(files=files))

        self.assertEqual(
            len(results),
            len(files),
            "There should be two failing results when files do not exist.",
        )

        for expected_file, result in zip(files, results):
            self.assertEqual(
                result.content,
                f"File does not exist: {expected_file}",
                f"Expected content for non-existing file {expected_file} does not match.",
            )
            self.assertEqual(
                result.metadata.get("file"),
                str(expected_file),
                f"Expected details for non-existing file {expected_file} do not match.",
            )

    @patch("pathlib.Path.exists", side_effect=[True, False])
    def test_execute__mixed_file_existence(self, mock_exists):
        files = [Path("file1.txt"), Path("file2.txt")]
        file_existence_validation = FileExistenceValidation()
        results, errors = asyncio.run(file_existence_validation.run(files=files))

        self.assertEqual(
            len(results), 2, "There should be two results when files have mixed existence."
        )
        self.assertEqual(
            len(errors), 0, "There should be no errors when files have mixed existence."
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
            results[1].content,
            f"File does not exist: {files[1]}",
            "Expected content for non-existing file does not match.",
        )
        self.assertEqual(
            results[1].metadata.get("file"),
            str(files[1]),
            "Expected details for non-existing file do not match.",
        )

    def test_execute__missing_files_parameter(self):
        file_existence_validation = FileExistenceValidation()
        with self.assertRaises(
            ValueError, msg="Should raise ValueError when 'files' parameter is missing."
        ):
            asyncio.run(file_existence_validation.run())
