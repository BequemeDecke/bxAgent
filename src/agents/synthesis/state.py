from langchain.agents.middleware import AgentState as BaseAgentState
from os import PathLike

class SynthesisAgentState(BaseAgentState):
    """
    State class for the synthesis agent. This class can be extended to include any additional state information that the synthesis agent may need to maintain during its operation.
    """
    written_files: list[PathLike] = []
