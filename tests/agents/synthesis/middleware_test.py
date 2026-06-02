import unittest
import logging

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain.agents import create_agent
from langchain.messages import ToolMessage, AIMessage, HumanMessage, ToolCall
from pathlib import Path

from bxagent.agents.synthesis.middleware import (
    extract_written_files,
    create_synthesis_response,
    SynthesisAgentStateMiddleware,
)

logger = logging.getLogger(__name__)


class TestSynthesisAgentMiddlewareFunctions(unittest.TestCase):
    """Test the extract_written_files and validate_file_paths functions of the SynthesisAgentStateMiddleware."""

    def setUp(self):
        self.WORKSPACE_PATH = Path("/test")
        self.UPDATED_FILE_INDEX = 13

    def test_extract_written_files_happy_path(self):
        """Test that extract_written_files correctly extracts file paths from a list of messages."""
        messages = [
            ToolMessage(
                name="write_file",
                content="Updated file /path/to/file1.txt",
                tool_call_id="call_1",
            ),
            ToolMessage(
                name="write_file",
                content="Updated file /path/to/file2.txt",
                tool_call_id="call_2",
            ),
            AIMessage(content="This is a message from the agent."),
        ]
        expected_file_paths = [
            self.WORKSPACE_PATH / "/path/to/file1.txt",
            self.WORKSPACE_PATH / "/path/to/file2.txt",
        ]
        extracted_file_paths = extract_written_files(
            messages, self.UPDATED_FILE_INDEX, self.WORKSPACE_PATH
        )
        self.assertEqual(extracted_file_paths, expected_file_paths)

    def test_extract_written_files_no_write_file_messages(self):
        """Test that extract_written_files returns an empty list when there are no write_file messages."""
        messages = [
            AIMessage(content="This is a message from the agent."),
            ToolMessage(
                name="some_other_tool",
                content="This is some other tool message.",
                tool_call_id="some_other_tool_call_id",
            ),
        ]
        expected_file_paths = []
        extracted_file_paths = extract_written_files(
            messages, self.UPDATED_FILE_INDEX, self.WORKSPACE_PATH
        )
        self.assertEqual(extracted_file_paths, expected_file_paths)

    def test_create_synthesis_response(self):
        """Test that create_synthesis_response correctly creates a structured response."""
        file_paths = [
            self.WORKSPACE_PATH / "file1.txt",
            self.WORKSPACE_PATH / "file2.txt",
        ]
        response = create_synthesis_response(file_paths)
        self.assertEqual(response.written_files, file_paths)


class TestSynthesisAgentMiddleware(unittest.TestCase):
    """Test the SynthesisAgentMiddleware by invoking it with a simple input and checking the output and state."""

    def setUp(self):
        self.UPDATED_FILE_INDEX = 13
        self.WORKSPACE_PATH = Path("/test")

    def test_middleware_processing(self):
        self.model = GenericFakeChatModel(
            messages=iter(
                [
                    "This is a message from the agent.",
                ]
            )
        )

        self.agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt="You are a helpful assistant that writes files.",
            middleware=[
                SynthesisAgentStateMiddleware(
                    updated_file_index=self.UPDATED_FILE_INDEX,
                    workspace_path=self.WORKSPACE_PATH,
                )
            ],
        )

        # Invoke the first time to get the ai message with the tool calls
        response = self.agent.invoke(
            input={
                "messages": [
                    HumanMessage(content="Write to file1.txt and file2.txt"),
                    ToolMessage(
                        name="write_file",
                        content="Updated file /path/to/file1.txt",
                        tool_call_id="call_1",
                    ),
                    ToolMessage(
                        name="write_file",
                        content="Updated file /path/to/file2.txt",
                        tool_call_id="call_2",
                    ),
                ],
            },
            version="v2",
        )

        # Check that the response is correct
        expected_file_paths = [
            self.WORKSPACE_PATH / "/path/to/file1.txt",
            self.WORKSPACE_PATH / "/path/to/file2.txt",
        ]

        expected_response = create_synthesis_response(expected_file_paths)
        actual_response = response.value.get("structured_response")

        self.assertEqual(actual_response, expected_response)
