from pathlib import Path
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

from bxagent.preparation.benchmarx import create_download_benchmarx_node
from bxagent.preparation.state import PreparationState


class TestBenchmarx(TestCase):
    def setUp(self):
        self.download_benchmarx = create_download_benchmarx_node()

    @patch("subprocess.run")
    def test_download_benchmarx(self, mock_run: Mock):
        mock_run.return_value.returncode = 0  # Simulate successful git clone

        input_state: PreparationState = {
            "workspace_path": Path("/path/to/workspace"),
            "install_benchmarx": True,
        }

        actual_state = self.download_benchmarx(input_state)

        mock_run.assert_called_once()
        self.assertIsNotNone(actual_state)
        self.assertEqual(
            actual_state.get("benchmarx_path"),
            input_state["workspace_path"] / "benchmarx",
        )

    @patch("subprocess.run")
    def test_download_benchmarx_skip(self, mock_run: Mock):
        input_state: PreparationState = {
            "workspace_path": Path("/path/to/workspace"),
            "install_benchmarx": False,
        }

        actual_state = self.download_benchmarx(input_state)
        self.assertIsNone(actual_state)  # Should return None when skipping installation

class TestBenchmarxIntegration(TestCase):
    def setUp(self):
        if shutil.which("git") is None:
            self.skipTest("Git is not available in the environment.")

    def test_download_benchmarx_integration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_path = Path(temp_dir, "workspace")
            workspace_path.mkdir()

            input_state: PreparationState = {
                "workspace_path": workspace_path,
                "install_benchmarx": True,
            }

            download_node = create_download_benchmarx_node()
            output_state = download_node(input_state)

            self.assertIsNotNone(output_state)
            benchmarx_path = output_state.get("benchmarx_path")
            self.assertIsNotNone(benchmarx_path)
            self.assertTrue(
                benchmarx_path.exists(),
                "The benchmarx path should point to an existing directory.",
            )
