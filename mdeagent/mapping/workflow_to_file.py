from mdeagent.state import MDEAgentState
from typing import Dict, Any


def map_workflow_to_file(state: MDEAgentState) -> Dict[str, Any]:
    return {"files": state.get("written_files", [])}
