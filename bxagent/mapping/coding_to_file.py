from bxagent.tools.coding.state import CodingAgentState
from typing import Dict, Any


def map_coding_to_file(state: CodingAgentState) -> Dict[str, Any]:
    return {"files": state.get("written_java_files", [])}
