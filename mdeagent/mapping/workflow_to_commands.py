from mdeagent.state import MDEAgentState
from typing import Dict, Any


def map_workflow_to_commands(state: MDEAgentState) -> Dict[str, Any]:
    return {"commands": state.get("required_commands", [])}
