"""
State management for the synthesis agent.

Uses a Node-style hook, because as mentioned in the documentation:
Run sequentially at specific execution points. Use for logging, validation, and state updates.

https://docs.langchain.com/oss/python/langchain/middleware/custom#node-style-hooks
"""

import logging

from langchain.messages import ToolMessage
from langchain.agents.middleware import (
    AgentState as BaseAgentState,
    AgentMiddleware,
)
from os import PathLike

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

        messages = state.get("messages", [])

        tool_messages: filter[ToolMessage] = filter(
            lambda msg: isinstance(msg, ToolMessage), messages
        )
        write_file_messages: filter[ToolMessage] = filter(
            lambda msg: msg.name == "write_file", tool_messages
        )
        written_files = [
            msg.content[UPDATED_FILE_INDEX:] for msg in write_file_messages
        ]
        logging.info(f"Agent has written the following files: {written_files}")
        return {"written_files": written_files}

    # def wrap_tool_call(
    #     self,
    #     request: ToolCallRequest,
    #     handler: Callable[[ToolCallRequest], ToolMessage | Command],
    # ) -> ToolMessage | Command:
    #     # Only track calls to the write_file tool
    #     tool_response = handler(request)

    #     if request.tool_call["name"] != "write_file":
    #         return tool_response

    #     file_path = request.tool_call["args"]["file_path"]
    #     files_in_state = request.state["written_files"]
    #     files = files_in_state + [file_path]
    #     logging.info(f"Tracking written file: {file_path}")
    #     return Command(update={"written_files": files})
