"""
State management for the synthesis agent.

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

from .output import SynthesisResponseFormat

UPDATED_FILE_INDEX = 13


class SynthesisAgentState(BaseAgentState):
    """
    State class for the synthesis agent. This class can be extended to include any additional state information that the synthesis agent may need to maintain during its operation.
    """

    written_files: list[PathLike] = []


class SynthesisAgentStateMiddleware(AgentMiddleware[SynthesisAgentState]):
    """
    Middleware for the synthesis agent that manages the state of the agent. This middleware can be used to track the files that have been written by the synthesis agent during its operation.
    """

    state_schema = SynthesisAgentState

    def after_agent(self, state: SynthesisAgentState, runtime):
        """
        After the agent has completed its execution,
        this method is called to update the state with the files that have been written by aggregating the calls to the write_file tool.

        The sad part: The tool messages don't store the args, so we have to parse the file path from the content of the tool message, which is in the format: "Updated file {file_path}".
        """

        def extract_written_files(messages: list[AnyMessage]):
            tool_messages: filter[ToolMessage] = filter(
                lambda msg: isinstance(msg, ToolMessage), messages
            )
            write_file_messages: filter[ToolMessage] = filter(
                lambda msg: msg.name == "write_file", tool_messages
            )
            return [msg.content[UPDATED_FILE_INDEX:] for msg in write_file_messages]

        messages = state.get("messages", [])
        written_files = extract_written_files(messages)

        logging.info(f"Agent has written the following files: {written_files}")
        structured_response = SynthesisResponseFormat(written_files=written_files)
        return {"structured_response": structured_response}
