from os import PathLike

from langchain.agents.middleware import (
    AgentState as BaseAgentState,
)


class CodingDeepAgentState(BaseAgentState):
    """
    State class for the Coding agent. This class can be extended to include any additional state information that the Coding agent may need to maintain during its operation.
    """

    written_files: list[PathLike] = []
