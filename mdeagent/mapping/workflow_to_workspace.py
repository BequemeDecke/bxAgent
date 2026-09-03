from mdeagent.state import MDEAgentState
from typing import Dict, Any


def map_workflow_to_workspace(state: MDEAgentState) -> Dict[str, Any]:
    return {
        "workspace_path": state["workspace_path"],
        "package_path": state["transformation_package_path"],
    }
