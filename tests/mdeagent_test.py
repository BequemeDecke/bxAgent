"""
This test file is the final test for the MDEAgent.

The inputs are some test files found in `./.mdeagent-tests/setup-files` and the required user input to create a maven package including the transformation class.
It uses the provided model in the .env file.
The output should be a functioning workspace. This should be checked by the tests.
"""

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest import TestCase

import pytest

from mdeagent.agent import build_mdeagent
from mdeagent.monitoring import build_langfuse_client
from mdeagent.state import MDEAgentState

logger = logging.getLogger(__name__)

TEST_ENVIRONMENT = Path(".mdeagent-tests")
TEST_SETUP_FILES = TEST_ENVIRONMENT / "setup-files"
TEST_EXECUTION_RUNS = TEST_ENVIRONMENT / "test-executions"


def create_workspace_folder() -> Path:
    workspace = TEST_EXECUTION_RUNS / datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    workspace.mkdir(parents=True)
    return workspace


class TestMDEAgent(TestCase):
    """Test case for the workflow architecture approach."""

    @pytest.fixture(autouse=True)
    def _setup_langfuse(self, enable_langfuse):
        """Set up Langfuse monitoring conditionally based on --enable-langfuse flag."""
        # Create a unique workspace for the test
        self.workspace_path = create_workspace_folder()
        logger.info(f"Created test workspace at {self.workspace_path}")

        # Check if the setup files exist
        self.source_model_path = TEST_SETUP_FILES / "Families"
        self.target_model_path = TEST_SETUP_FILES / "Persons"
        if not self.source_model_path.exists() or not self.target_model_path.exists():
            self.fail(
                f"Setup files not found. Please ensure that {self.source_model_path} and {self.target_model_path} exist."
            )
        if len(list(self.source_model_path.glob("*.java"))) != 4:
            self.fail(
                f"Expected 4 source model files in {self.source_model_path}, but found {len(list(self.source_model_path.glob('*.java')))}."
            )
        if len(list(self.target_model_path.glob("*.java"))) != 3:
            self.fail(
                f"Expected 3 target model files in {self.target_model_path}, but found {len(list(self.target_model_path.glob('*.java')))}."
            )

        # Build the workflow agent without BenchmarX support
        self.agent = build_mdeagent(self.workspace_path, benchmarx_path=None).compile()

        # Build the Langfuse client for monitoring (optional)
        self.enable_langfuse = enable_langfuse
        if enable_langfuse:
            self.langfuse_client, self.langfuse_callback_handler = (
                build_langfuse_client()
            )
        else:
            self.langfuse_client = None
            self.langfuse_callback_handler = None

    def test_mdeagent_workflow(self):
        """The test method for the MDEAgent workflow.

        Note: Only ainvoke can be used here, because some nodes are executed asynchronously and the test needs to wait for them to finish. The test will fail if the workflow is not completed successfully.
        """
        # 1. Create the initial state for the agent
        initial_state = MDEAgentState(
            source_model_path=self.source_model_path,
            target_model_path=self.target_model_path,
            group_id="de.hofuniversity",
            artifact_id="MDEAgentFamilyToPerson",
            required_commands=[
                "mvn",
                "java",
                "javac",
                "jar",
            ],  # This should not be set by the user
        )

        # 2. Invoke the agent with the initial state
        callbacks = (
            [self.langfuse_callback_handler] if self.langfuse_callback_handler else []
        )
        output = asyncio.run(
            self.agent.ainvoke(
                initial_state, config={"callbacks": callbacks}, version="v2"
            )
        )
        if self.langfuse_client:
            self.langfuse_client.flush()

        # 3. Check the output state for expected values
        self.check_output_state(output)

        # 4. Check the contents of the workspace for expected files
        self.check_workspace_contents()

    def check_output_state(self, output: MDEAgentState):
        """Check the output state for expected values."""
        # Check that the transformation class path is set
        self.assertIsNotNone(
            output.get("transformation_class_path"),
            "Transformation class path should not be None.",
        )
        self.assertTrue(
            output["transformation_class_path"].exists(),
            "Transformation class file does not exist.",
        )

        # Check that the bxtool path is set
        self.assertIsNotNone(
            output.get("bxtool_path"), "BXT tool path should not be None."
        )
        self.assertTrue(output["bxtool_path"].exists(), "BXT tool file does not exist.")

        # Check that the written files list is not empty
        self.assertGreater(
            len(output.get("written_files", [])),
            0,
            "No files were written by the implementation node.",
        )

        # Check that the latest evaluation runs list is not empty
        self.assertGreater(
            len(output.get("latest_evaluation_runs", [])),
            0,
            "No evaluation runs were recorded.",
        )

    def check_workspace_contents(self):
        """Check the contents of the workspace for expected files."""
        # Check that the workspace contains the expected files
        expected_files = ["transformation_class.java", "bxtool.jar"]
        for file_name in expected_files:
            file_path = self.workspace_path / file_name
            self.assertTrue(
                file_path.exists(),
                f"Expected file {file_name} does not exist in the workspace.",
            )
