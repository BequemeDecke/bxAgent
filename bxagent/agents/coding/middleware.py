"""
State management for the Coding agent.

Uses a Node-style hook, because as mentioned in the documentation:
Run sequentially at specific execution points. Use for logging, validation, and state updates.

https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks
"""

import logging

from langchain.messages import AnyMessage, ToolMessage
from langchain.agents.middleware import (
    AgentState as BaseAgentState,
    AgentMiddleware,
)
from os import PathLike
from pathlib import Path

from bxagent.config import Config

UPDATED_FILE_INDEX = Config.get_instance().VARIABLES.UPDATED_FILE_INDEX
WORKSPACE_PATH = Config.get_instance().WORKSPACE.PATH


class CodingDeepAgentState(BaseAgentState):
    """
    State class for the Coding agent. This class can be extended to include any additional state information that the Coding agent may need to maintain during its operation.
    """

    written_files: list[PathLike] = []


def extract_written_files(
    messages: list[AnyMessage],
    updated_file_index: int = UPDATED_FILE_INDEX,
    workspace_path: Path = WORKSPACE_PATH,
) -> list[Path]:
    """
    Extract the paths of files that have been written by the Coding agent from a list of messages.
    """

    tool_messages: filter[ToolMessage] = filter(
        lambda msg: isinstance(msg, ToolMessage), messages
    )
    write_file_messages: filter[ToolMessage] = filter(
        lambda msg: msg.name == "write_file", tool_messages
    )
    relative_file_paths: map[str] = map(
        lambda msg: msg.content[updated_file_index:], write_file_messages
    )
    absolute_file_paths: map[Path] = map(
        lambda path: workspace_path / path, relative_file_paths
    )
    return list(absolute_file_paths)


class CodingDeepAgentStateMiddleware(AgentMiddleware[CodingDeepAgentState]):
    """
    Middleware for the Coding agent that manages the state of the agent. This middleware can be used to track the files that have been written by the Coding agent during its operation.
    """

    state_schema = CodingDeepAgentState

    def __init__(
        self,
        updated_file_index: int = UPDATED_FILE_INDEX,
        workspace_path: Path = WORKSPACE_PATH,
    ):
        self.updated_file_index = updated_file_index
        self.workspace_path = workspace_path

    def after_agent(self, state: CodingDeepAgentState, runtime):
        """
        After the agent has completed its execution,
        this method is called to update the state with the files that have been written by aggregating the calls to the write_file tool.

        The sad part: The tool messages don't store the args, so we have to parse the file path from the content of the tool message, which is in the format: "Updated file {file_path}".
        """
        messages = state.get("messages", [])
        written_files = extract_written_files(
            messages, self.updated_file_index, self.workspace_path
        )

        logging.info(f"Agent has written the following files: {written_files}")
        return {"written_files": written_files}
