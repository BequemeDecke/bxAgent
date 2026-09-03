import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from mdeagent.implementation.format_code import create_format_code_node
from mdeagent.implementation.state import ImplementationState


class TestFormatCodeNode(TestCase):
    def setUp(self):
        self.format_code_node = create_format_code_node(workspace=Path("/fake/workspace"))

    @patch("mdeagent.implementation.format_code.format_java_files")
    def test_format_code_node__calls_format_java_files(self, mock_format):
        """Test that the format_code node calls format_java_files"""
        state: ImplementationState = {
            "transformation_md": None,  # type: ignore
            "task_specification": "Test task",
            "written_java_files": [],
            "bxtool_path": Path("/fake/path"),
            "transformation_implementation": "test implementation",
            "latest_evaluation_results": {},
            "implementation_iteration": 1,
        }

        result = self.format_code_node(state)

        mock_format.assert_called_once_with(Path("/fake/workspace"))
        self.assertEqual(result, state)


class TestFormatJavaFiles(TestCase):
    @patch("subprocess.run")
    def test_format_java_files__success(self, mock_run):
        """Test that format_java_files runs successfully when mvn spotless:apply succeeds"""
        from mdeagent.preparation.pom import format_java_files

        mock_run.return_value = subprocess.CompletedProcess(
            args=["mvn", "spotless:apply"], returncode=0, stdout="", stderr=""
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            format_java_files(workspace)

            mock_run.assert_called_once_with(
                ["mvn", "spotless:apply"],
                cwd=workspace,
                capture_output=True,
                text=True,
            )

    @patch("subprocess.run")
    def test_format_java_files__failure_raises_error(self, mock_run):
        """Test that format_java_files raises RuntimeError when formatting fails"""
        from mdeagent.preparation.pom import format_java_files

        mock_run.return_value = subprocess.CompletedProcess(
            args=["mvn", "spotless:apply"],
            returncode=1,
            stdout="Some output",
            stderr="Some error",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            with self.assertRaises(RuntimeError) as context:
                format_java_files(workspace)

            self.assertIn("Failed to format Java files", str(context.exception))
            self.assertIn("Return code: 1", str(context.exception))
